"""Fixed spatial sample of HLS v2.0 COG blocks plus a per-season predicted-corn mask.

Why a sample
------------
HLS (Harmonized Landsat Sentinel-2, v2.0; Planetary Computer ``hls2-s30`` /
``hls2-l30``) is 30 m on the Sentinel-2 MGRS tile grid: UTM tiles of 3660 x 3660 px
(109.8 km), stepping 100 km, so neighbours overlap by 9.8 km. Loading whole states at
30 m for every scene is too expensive, so we read a fixed set of internal COG blocks
(256 x 256 px = 7.68 km, verified with rasterio ``block_shapes`` on ``hls2-s30`` B04)
that sit on corn-heavy ground, and mask them per season with a rotation-predicted corn
layer.

Pipeline
--------
1. ``load_tiles``: MGRS tiles intersecting the five core states (tile id, EPSG, 30 m
   transform) discovered from ``hls2-s30`` STAC items; cached ``tiles.parquet``.
2. Candidate blocks: COG-block-aligned windows (``row_off``/``col_off`` multiples of
   ``BLOCK``) inside each tile's *core* (the first ``CORE_PX`` rows/cols = 100 km, i.e.
   without the overlap margin shared with the east/south neighbours), whose EPSG:5070
   footprint lies fully inside the state polygon shrunk by ``STATE_INSET_M``.
3. Scoring: mean CDL corn fraction over **CDL 2008-2012 only** (``SELECT_CDL_YEARS``)
   sampled from the cached MODIS-grid corn fraction rasters. These years pre-date every
   HLS season used (including the 2013-2015 climatology presample), so the choice of
   blocks cannot leak information about any modelled season.
4. Selection (greedy, deterministic): per state, ``TILES_PER_STATE`` tiles by corn mass
   with a minimum centre spacing that relaxes through ``TILE_SPACINGS_M`` until enough
   tiles are found; then the ``BLOCKS_PER_TILE`` best blocks per tile, skipping any
   whose EPSG:5070 footprint overlaps an already-selected block (tile overlaps and UTM
   zone seams). Output ``blocks.parquet``.
5. Season mask (no look-ahead; CDL Y-1 is released ~Jan-Feb of Y): a pixel is predicted
   corn in season Y if (CDL Y-1 == soybean and CDL Y-2 == corn) or (corn in all of
   Y-1, Y-2, Y-3), followed by a 1-pixel binary erosion (3x3) to drop field edges. CDL
   (EPSG:5070) is resampled into each block's UTM 30 m grid with nearest neighbour,
   reading only a small window (``cornmask.cdl_codes_window``) with a 1-pixel margin so
   the erosion is exact at block edges. Cached per state-season as packed bits.

``mask_purity`` is an ex-post *diagnostic only* (uses CDL of the season itself) and must
never feed a feature.
"""

from __future__ import annotations

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from affine import Affine
from pyproj import Transformer
from shapely.geometry import box

from cornsat import cornmask
from cornsat.config import DATA_INTERIM, STATES

HLS_DIR = DATA_INTERIM / "hls"

# --- grid constants -----------------------------------------------------------------
HLS_RES = 30.0
TILE_PX = 3660  # 109.8 km
TILE_STEP_M = 100_000.0  # MGRS 100 km squares; tiles overlap neighbours by 9.8 km
BLOCK = 256  # internal COG block of HLS v2.0 assets on Planetary Computer (verified)
CORE_PX = int(TILE_STEP_M // HLS_RES)  # 3333: rows/cols of the non-overlapping core
EQUAL_AREA = "EPSG:5070"

# --- selection constants (module-level so the pilot can resize them) ----------------
TILES_PER_STATE = 8
BLOCKS_PER_TILE = 9
SELECT_CDL_YEARS = (2008, 2009, 2010, 2011, 2012)
STATE_INSET_M = 1000.0  # 1:20M Census polygons are generalised; stay clear of borders
TILE_SPACINGS_M = (150_000.0, 110_000.0, 0.0)  # min tile-centre spacing, relaxed in turn
# A tile is eligible only if the mean score of its best BLOCKS_PER_TILE blocks is at
# least this share of the state's best tile (keeps spacing from forcing corn-free tiles).
TILE_MIN_REL_CORN = 0.5

# --- mask constants -----------------------------------------------------------------
MARGIN = 1  # extra CDL pixels around each block so the erosion is exact at its edges
CORN = cornmask.CORN_CLASS
SOY = 5
MASK_RULE = (
    "corn(Y) := (CDL[Y-1]==soy & CDL[Y-2]==corn) | (CDL[Y-1]==CDL[Y-2]==CDL[Y-3]==corn); "
    "then 3x3 binary erosion"
)
STAC_URL = cornmask.STAC_URL
TILE_SEARCH_WINDOWS = ("2023-07-01/2023-07-31", "2023-08-01/2023-08-31", "2022-07-01/2022-07-31")
N_WORKERS = 8
_WARP_LOCK = threading.Lock()


# =====================================================================================
# geometry helpers
# =====================================================================================
@lru_cache(maxsize=16)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(src, dst, always_xy=True)


def _reproject_geom(geom, src: str, dst: str, seg: float | None = None):
    """Reproject a shapely geometry (optionally densified first, ``seg`` in source units)."""
    if seg:
        geom = shapely.segmentize(geom, seg)
    tr = _transformer(src, dst)

    def f(xy):
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geom, f)


def _epsg_str(epsg: int) -> str:
    return f"EPSG:{int(epsg)}"


@lru_cache(maxsize=16)
def _state_geom_5070(state: str):
    return cornmask._state_geom_4326(state).to_crs(EQUAL_AREA).geometry.iloc[0]


@lru_cache(maxsize=16)
def _state_inset_5070(state: str):
    g = _state_geom_5070(state).buffer(-STATE_INSET_M)
    shapely.prepare(g)
    return g


def block_transform(ulx: float, uly: float, row_off: int, col_off: int) -> Affine:
    """30 m transform of a block window whose tile upper-left corner is (ulx, uly)."""
    return Affine(HLS_RES, 0.0, ulx + col_off * HLS_RES, 0.0, -HLS_RES, uly - row_off * HLS_RES)


def core_offsets() -> list[int]:
    """Block-aligned offsets whose whole block fits inside the tile core."""
    return list(range(0, CORE_PX - BLOCK + 1, BLOCK))


# =====================================================================================
# 1. MGRS tile table
# =====================================================================================
def _tiles_path() -> Path:
    return HLS_DIR / "tiles.parquet"


def _tile_row(tile: str, epsg: int, transform) -> dict:
    t = Affine(*transform[:6])
    if abs(t.a - HLS_RES) > 1e-6 or abs(t.e + HLS_RES) > 1e-6:
        raise ValueError(f"unexpected HLS transform for {tile}: {t}")
    ulx, uly = t.c, t.f
    fp_utm = box(ulx, uly - TILE_PX * HLS_RES, ulx + TILE_PX * HLS_RES, uly)
    fp = _reproject_geom(fp_utm, _epsg_str(epsg), EQUAL_AREA, seg=5000.0)
    return {
        "tile": tile,
        "epsg": int(epsg),
        "ulx": float(ulx),
        "uly": float(uly),
        "transform": [float(v) for v in tuple(t)[:6]],
        "footprint_5070": shapely.to_wkb(fp),
    }


def _discover_tiles(states) -> pd.DataFrame:
    import planetary_computer as pc
    import pystac_client

    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    rows: dict[str, dict] = {}
    need = shapely.union_all([_state_geom_5070(s) for s in states])
    for window in TILE_SEARCH_WINDOWS:
        for st in states:
            geom = cornmask._state_geom_4326(st).geometry.iloc[0].simplify(0.02)
            search = cat.search(
                collections=["hls2-s30"],
                intersects=shapely.geometry.mapping(geom.envelope),
                datetime=window,
                limit=500,
            )
            for it in search.items():
                tile = it.id.split(".")[2]
                if tile in rows:
                    continue
                code = it.properties.get("proj:code") or f"EPSG:{it.properties['proj:epsg']}"
                rows[tile] = _tile_row(tile, int(code.split(":")[1]), it.properties["proj:transform"])
        cover = shapely.union_all([shapely.from_wkb(r["footprint_5070"]) for r in rows.values()])
        if cover.buffer(1.0).contains(need):
            break
    else:
        raise RuntimeError("HLS tiles found do not cover all requested states")
    df = pd.DataFrame(sorted(rows.values(), key=lambda r: r["tile"]))
    fps = shapely.from_wkb(df["footprint_5070"].values)
    df["states"] = [
        ",".join(s for s in states if fp.intersects(_state_geom_5070(s))) for fp in fps
    ]
    return df[df["states"] != ""].reset_index(drop=True)


def load_tiles(refresh: bool = False) -> pd.DataFrame:
    """MGRS/HLS tiles intersecting the core states (cached ``tiles.parquet``)."""
    path = _tiles_path()
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    df = _discover_tiles(list(STATES))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


# =====================================================================================
# 2. candidate blocks and scoring
# =====================================================================================
def candidate_blocks(tile: pd.Series | dict, state: str, inset_geom=None) -> pd.DataFrame:
    """Block-aligned windows in the tile core whose EPSG:5070 footprint is inside the state.

    ``inset_geom`` (EPSG:5070) defaults to the state polygon shrunk by STATE_INSET_M.
    """
    inset = _state_inset_5070(state) if inset_geom is None else inset_geom
    epsg, ulx, uly = int(tile["epsg"]), float(tile["ulx"]), float(tile["uly"])
    offs = core_offsets()
    rr, cc = np.meshgrid(offs, offs, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    x0 = ulx + cc * HLS_RES
    y1 = uly - rr * HLS_RES
    boxes_utm = shapely.box(x0, y1 - BLOCK * HLS_RES, x0 + BLOCK * HLS_RES, y1)
    fps = np.array(
        [_reproject_geom(b, _epsg_str(epsg), EQUAL_AREA, seg=BLOCK * HLS_RES / 4) for b in boxes_utm]
    )
    inside = shapely.contains(inset, fps)
    tid = tile["tile"]
    df = pd.DataFrame(
        {
            "block_id": [f"{state}_{tid}_r{r:04d}_c{c:04d}" for r, c in zip(rr, cc)],
            "state": state,
            "tile": tid,
            "epsg": epsg,
            "row_off": rr.astype(int),
            "col_off": cc.astype(int),
            "height": BLOCK,
            "width": BLOCK,
            "ulx": ulx,
            "uly": uly,
            "geometry": fps,
        }
    )
    return df[inside].reset_index(drop=True)


def _selection_weight(state: str):
    """Mean MODIS-grid CDL corn fraction over SELECT_CDL_YEARS only (2008-2012)."""
    years = list(SELECT_CDL_YEARS)
    w = cornmask.season_weight(state, years[-1] + 1, cdl_years=years)
    if [int(y) for y in w.attrs["cdl_years"]] != years:
        raise AssertionError("block selection must use CDL 2008-2012 only")
    gb = cornmask.modis_geobox(state)
    return np.asarray(w.values, dtype="float64"), gb.affine, cornmask.MODIS_PROJ


def score_candidates(cands: pd.DataFrame, weight: np.ndarray, affine: Affine, crs: str) -> np.ndarray:
    """Mean of ``weight`` over the pixels whose centres fall inside each candidate block."""
    out = np.full(len(cands), np.nan)
    if len(cands) == 0:
        return out
    h, w = weight.shape
    jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xs = affine.c + jj * affine.a + ii * affine.b
    ys = affine.f + jj * affine.d + ii * affine.e
    vals = weight.ravel()
    nb = len(core_offsets())
    for epsg, grp_e in cands.groupby("epsg"):
        ux, uy = _transformer(crs, _epsg_str(epsg)).transform(xs.ravel(), ys.ravel())
        ux, uy = np.asarray(ux), np.asarray(uy)
        for _, grp in grp_e.groupby("tile"):
            ulx, uly = float(grp["ulx"].iloc[0]), float(grp["uly"].iloc[0])
            col = np.floor((ux - ulx) / HLS_RES)
            row = np.floor((uly - uy) / HLS_RES)
            ok = (col >= 0) & (row >= 0) & (col < nb * BLOCK) & (row < nb * BLOCK)
            bi = (row[ok] // BLOCK).astype(int) * nb + (col[ok] // BLOCK).astype(int)
            s = np.bincount(bi, weights=vals[ok], minlength=nb * nb)
            n = np.bincount(bi, minlength=nb * nb)
            idx = (grp["row_off"].values // BLOCK) * nb + grp["col_off"].values // BLOCK
            with np.errstate(invalid="ignore", divide="ignore"):
                out[grp.index.values] = s[idx] / n[idx]
    return out


# =====================================================================================
# 3. greedy selection
# =====================================================================================
def _overlaps_any(geom, chosen: list) -> bool:
    for g in chosen:
        if geom.intersects(g) and geom.intersection(g).area > 1.0:  # m2; touching is fine
            return True
    return False


def select_blocks(
    cands: pd.DataFrame,
    states=None,
    tiles_per_state: int | None = None,
    blocks_per_tile: int | None = None,
    spacings=None,
) -> pd.DataFrame:
    """Greedy deterministic selection from scored candidates (columns as candidate_blocks
    plus ``score``). Tiles per state by corn mass (sum of candidate scores) among tiles
    whose best-``bpt`` mean score is >= TILE_MIN_REL_CORN x the state's best, with a
    minimum centre spacing relaxed through ``spacings``; then the best blocks per tile;
    no two selected EPSG:5070 footprints overlap (touching edges allowed)."""
    tps = TILES_PER_STATE if tiles_per_state is None else tiles_per_state
    bpt = BLOCKS_PER_TILE if blocks_per_tile is None else blocks_per_tile
    spacings = TILE_SPACINGS_M if spacings is None else spacings
    states = list(dict.fromkeys(cands["state"])) if states is None else list(states)
    cands = cands[np.isfinite(cands["score"])]
    chosen_geoms: list = []
    out = []
    for st in states:
        cs = cands[cands["state"] == st]
        if cs.empty:
            continue
        g = cs.groupby("tile")
        mass = g["score"].sum()
        nblk = g.size()
        topk = g["score"].apply(lambda s: s.nlargest(bpt).mean())
        full = [t for t in mass.index if nblk[t] >= bpt]
        if not full:
            continue
        floor = TILE_MIN_REL_CORN * max(topk[t] for t in full)
        eligible = sorted(
            [t for t in full if topk[t] >= floor], key=lambda t: (-mass[t], t)
        )
        centres = {}
        for t in eligible:
            u = shapely.union_all(cs.loc[cs["tile"] == t, "geometry"].values)
            centres[t] = u.centroid
        tiles: list[str] = []
        for sp in spacings:
            for t in eligible:
                if len(tiles) >= tps:
                    break
                if t in tiles:
                    continue
                if all(centres[t].distance(centres[u]) >= sp for u in tiles):
                    tiles.append(t)
            if len(tiles) >= tps:
                break
        rank = 0
        for tile_rank, t in enumerate(tiles, start=1):
            ct = cs[cs["tile"] == t].sort_values(
                ["score", "row_off", "col_off"], ascending=[False, True, True]
            )
            n = 0
            for _, r in ct.iterrows():
                if n >= bpt:
                    break
                if _overlaps_any(r["geometry"], chosen_geoms):
                    continue
                chosen_geoms.append(r["geometry"])
                rank += 1
                n += 1
                rec = r.to_dict()
                rec.update(tile_rank=tile_rank, tile_mass=float(mass[t]), rank=rank)
                out.append(rec)
    return pd.DataFrame(out)


# =====================================================================================
# blocks table
# =====================================================================================
def _blocks_path() -> Path:
    return HLS_DIR / "blocks.parquet"


def _finalise(sel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in sel.iterrows():
        t = block_transform(r["ulx"], r["uly"], r["row_off"], r["col_off"])
        x0, y1 = t.c, t.f
        rows.append(
            {
                "block_id": r["block_id"],
                "state": r["state"],
                "tile": r["tile"],
                "epsg": int(r["epsg"]),
                "row_off": int(r["row_off"]),
                "col_off": int(r["col_off"]),
                "height": int(r["height"]),
                "width": int(r["width"]),
                "transform": [float(v) for v in tuple(t)[:6]],
                "bounds_utm": [
                    x0,
                    y1 - r["height"] * HLS_RES,
                    x0 + r["width"] * HLS_RES,
                    y1,
                ],
                "geometry_5070": shapely.to_wkb(r["geometry"]),
                "corn_frac_0812": float(r["score"]),
                "rank": int(r["rank"]),
                "tile_rank": int(r["tile_rank"]),
            }
        )
    return pd.DataFrame(rows)


def build_blocks(refresh: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Build (or load) the block sample for all core states (selection is joint)."""
    path = _blocks_path()
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    tiles = load_tiles()
    parts = []
    for st in STATES:
        t0 = time.time()
        tt = tiles[tiles["states"].str.split(",").apply(lambda s: st in s)]
        cands = pd.concat(
            [candidate_blocks(r, st) for _, r in tt.iterrows()], ignore_index=True
        )
        weight, aff, crs = _selection_weight(st)
        cands["score"] = score_candidates(cands, weight, aff, crs)
        parts.append(cands)
        if verbose:
            print(
                f"{st}: {len(tt)} tiles, {len(cands)} candidate blocks, "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
    sel = select_blocks(pd.concat(parts, ignore_index=True), states=list(STATES))
    df = _finalise(sel)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def load_blocks(states=None) -> pd.DataFrame:
    """The fixed block sample (builds it on first use). ``states``: abbr or list."""
    df = build_blocks(verbose=False)
    if states is not None:
        sts = [states] if isinstance(states, str) else list(states)
        df = df[df["state"].isin(sts)]
    return df.reset_index(drop=True)


# =====================================================================================
# 4. CDL in block grids and the rotation-predicted corn mask
# =====================================================================================
def _margin_grid(row) -> tuple[Affine, tuple[int, int]]:
    t = Affine(*row["transform"])
    aff = Affine(t.a, t.b, t.c - MARGIN * t.a, t.d, t.e, t.f - MARGIN * t.e)
    return aff, (int(row["height"]) + 2 * MARGIN, int(row["width"]) + 2 * MARGIN)


def _block_codes(row, cdl_year: int) -> np.ndarray:
    """CDL class codes for one block (+MARGIN) resampled (nearest) to its UTM 30 m grid."""
    from rasterio.warp import Resampling, reproject, transform_bounds

    aff, (h, w) = _margin_grid(row)
    crs = _epsg_str(row["epsg"])
    b = transform_bounds(crs, EQUAL_AREA, aff.c, aff.f - h * HLS_RES, aff.c + w * HLS_RES, aff.f,
                         densify_pts=21)
    pad = 2 * cornmask.CDL_RES
    bounds = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
    codes, src_aff = cornmask.cdl_codes_window(cdl_year, bounds, row["state"])
    dst = np.zeros((h, w), dtype="uint8")
    with _WARP_LOCK:  # network reads run in parallel; the tiny in-memory warp does not
        reproject(
            source=codes,
            destination=dst,
            src_transform=src_aff,
            src_crs=EQUAL_AREA,
            dst_transform=aff,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )
    return dst


def _cdl_path(state: str, cdl_year: int) -> Path:
    return HLS_DIR / "cdl" / f"{state}_{cdl_year}.npz"


def block_cdl(state: str, cdl_year: int, refresh: bool = False) -> dict[str, np.ndarray]:
    """CDL codes (uint8, block grid + MARGIN) for every sampled block of a state; cached."""
    blocks = load_blocks(state)
    ids = list(blocks["block_id"])
    path = _cdl_path(state, cdl_year)
    if path.exists() and not refresh:
        with np.load(path) as z:
            if set(ids) <= set(z.files):
                return {k: z[k] for k in ids}
    rows = [r for _, r in blocks.iterrows()]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        arrs = list(ex.map(lambda r: _block_codes(r, cdl_year), rows))
    out = dict(zip(ids, arrs))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(path)
    return out


def season_cdl_years(season: int) -> list[int]:
    """CDL years [Y-1, Y-2, Y-3] (clipped to >= CDL_FIRST_YEAR) used by the season-Y mask."""
    if season - 1 > cornmask.CDL_LAST_YEAR:
        raise ValueError(f"season {season} needs CDL {season - 1} (latest {cornmask.CDL_LAST_YEAR})")
    if season - 2 < cornmask.CDL_FIRST_YEAR:
        raise ValueError(f"season {season} needs CDL {season - 2} (first {cornmask.CDL_FIRST_YEAR})")
    return [y for y in (season - 1, season - 2, season - 3) if y >= cornmask.CDL_FIRST_YEAR]


def rotation_corn(prev1: np.ndarray, prev2: np.ndarray, prev3: np.ndarray | None = None) -> np.ndarray:
    """Predicted corn from CDL codes of Y-1, Y-2 (and Y-3 if available)."""
    rot = (prev1 == SOY) & (prev2 == CORN)
    cont = (prev1 == CORN) & (prev2 == CORN)
    if prev3 is not None:
        cont &= prev3 == CORN
    return rot | cont


def erode(mask: np.ndarray) -> np.ndarray:
    """1-pixel binary erosion with a 3x3 structuring element (pixels beyond edge = False)."""
    from scipy.ndimage import binary_erosion

    return binary_erosion(mask, structure=np.ones((3, 3), bool), border_value=0)


def _compute_masks(state: str, season: int) -> tuple[dict[str, np.ndarray], list[int]]:
    years = season_cdl_years(season)
    assert all(y < season for y in years)
    stacks = [block_cdl(state, y) for y in years]
    out = {}
    for bid in stacks[0]:
        layers = [s[bid] for s in stacks]
        pred = erode(rotation_corn(*layers))
        out[bid] = pred[MARGIN:-MARGIN, MARGIN:-MARGIN] if MARGIN else pred
    return out, years


def _mask_path(state: str, season: int) -> Path:
    return HLS_DIR / "masks" / f"{state}_{season}.npz"


def build_masks(season: int, states=None, refresh: bool = False) -> dict[str, Path]:
    """Build/cache the season mask of every sampled block, one npz per state-season.

    Arrays are keyed by block_id and stored as ``np.packbits`` of the flattened (h, w)
    bool mask; ``__meta__`` holds a JSON string with cdl_years, shapes and the rule.
    """
    sts = list(STATES) if states is None else ([states] if isinstance(states, str) else list(states))
    paths = {}
    for st in sts:
        path = _mask_path(st, season)
        paths[st] = path
        if path.exists() and not refresh:
            continue
        masks, years = _compute_masks(st, season)
        meta = {
            "state": st,
            "season": int(season),
            "cdl_years": years,
            "rule": MASK_RULE,
            "shapes": {k: list(v.shape) for k, v in masks.items()},
        }
        arrays = {k: np.packbits(v.ravel()) for k, v in masks.items()}
        arrays["__meta__"] = np.array(json.dumps(meta))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.replace(path)
        _load_masks.cache_clear()
    return paths


@lru_cache(maxsize=32)
def _load_masks(path_str: str, mtime: float) -> tuple[dict, dict]:
    with np.load(path_str) as z:
        meta = json.loads(str(z["__meta__"]))
        packed = {k: z[k] for k in z.files if k != "__meta__"}
    return packed, meta


def _masks_for(state: str, season: int) -> tuple[dict, dict]:
    path = _mask_path(state, season)
    if not path.exists():
        build_masks(season, [state])
    return _load_masks(str(path), path.stat().st_mtime)


def mask_meta(state: str, season: int) -> dict:
    """Metadata of a cached state-season mask (cdl_years used, rule, shapes)."""
    return _masks_for(state, season)[1]


def season_mask(block_id: str, season: int) -> np.ndarray:
    """Predicted-corn mask (bool, (height, width)) aligned to the block's window in its tile."""
    state = block_id.split("_", 1)[0]
    packed, meta = _masks_for(state, season)
    if block_id not in packed:
        raise KeyError(f"{block_id} not in mask cache for {state} {season}")
    h, w = meta["shapes"][block_id]
    return np.unpackbits(packed[block_id], count=h * w).reshape(h, w).astype(bool)


def mask_purity(state: str, season: int) -> float:
    """DIAGNOSTIC ONLY (ex post, never a feature): share of predicted-corn pixels that CDL
    of ``season`` itself labels corn, pooled over the state's blocks."""
    if season > cornmask.CDL_LAST_YEAR:
        raise ValueError(f"CDL {season} not available (latest {cornmask.CDL_LAST_YEAR})")
    packed, meta = _masks_for(state, season)
    truth = block_cdl(state, season)
    hit = tot = 0
    for bid in packed:
        m = season_mask(bid, season)
        t = truth[bid]
        t = t[MARGIN:-MARGIN, MARGIN:-MARGIN] if MARGIN else t
        hit += int((m & (t == CORN)).sum())
        tot += int(m.sum())
    return hit / tot if tot else math.nan

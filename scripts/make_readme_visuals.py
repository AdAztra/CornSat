"""Build the README images in docs/img/ from real satellite data.

    uv run python scripts/make_readme_visuals.py                 # everything
    uv run python scripts/make_readme_visuals.py --only hero     # one image
    names: hero resolution sampling_map derecho season results

Idempotent. Downloads (HLS block windows through the Earthdata token handled by
``cornsat.hls``; MODIS from Planetary Computer, no key) are cached as small ``.npz`` files in
``--cache`` (default: ``$README_CACHE`` or a folder under the system temp dir), so reruns
need no network. Nothing is written to ``data/``. Imagery is public domain (NASA HLS,
USDA CDL, MODIS); see the README credits.
"""

from __future__ import annotations

import argparse
import io
import os
import tempfile
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import gridspec  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from PIL import Image  # noqa: E402
from pyproj import Transformer  # noqa: E402

from cornsat import hls, hlssample  # noqa: E402
from cornsat.config import ROOT, STATES  # noqa: E402

OUT = ROOT / "docs" / "img"
CACHE = Path(os.environ.get("README_CACHE", Path(tempfile.gettempdir()) / "cornsat_readme_cache"))

# --- what is shown (chosen from data/interim/hls/scenes: clear scenes, corn-rich blocks) ---
HERO = dict(block="IA_T15TVH_r2304_c2304", day="2023-07-30", sensor="S30")
SEASON = dict(block="IA_T15TVH_r2304_c2304", year=2023)
DERECHO = dict(
    block="IA_T15TXG_r0256_c0512",
    before=("2020-07-27", "S30"),
    after=("2020-08-21", "S30"),
)  # the derecho crossed Iowa on 2020-08-10
# one of the two NDVI colour scales used everywhere
NDVI_CMAP = plt.get_cmap("YlGn").copy()
NDVI_CMAP.set_bad("#bdbdbd")  # cloud / shadow / no data
NDVI_KW = dict(cmap=NDVI_CMAP, vmin=0.3, vmax=0.95)
GREY = "#9a9a9a"
INK = "#1f2a1f"
PNG_DPI = 110


# =====================================================================================
# data access (cached)
# =====================================================================================
def _blocks() -> pd.DataFrame:
    return hlssample.load_blocks().set_index("block_id", drop=False)


def block_geo(block_id: str) -> dict:
    r = _blocks().loc[block_id]
    x0, y0, x1, y1 = (float(v) for v in r["bounds_utm"])
    lon, lat = Transformer.from_crs(int(r["epsg"]), 4326, always_xy=True).transform(
        (x0 + x1) / 2, (y0 + y1) / 2
    )
    return dict(row=r, epsg=int(r["epsg"]), bounds=(x0, y0, x1, y1), lon=lon, lat=lat,
                tile=r["tile"], state=r["state"])


def _items(year: int, tile: str) -> pd.DataFrame:
    p = hls._items_path(year, tile)
    if not p.exists():
        raise FileNotFoundError(f"granule list {p.name} missing; run scripts/build_hls_sample.py")
    return pd.read_parquet(p)


def _ll(lat: float, lon: float) -> str:
    return f"{abs(lat):.2f}°{'N' if lat >= 0 else 'S'} {abs(lon):.2f}°{'W' if lon < 0 else 'E'}"


def hls_scene(block_id: str, day: str, sensor: str) -> dict:
    """True-colour bands, NDVI and Fmask of a block on a day (cached, 256x256)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"hls_{block_id}_{day}_{sensor}.npz"
    if not path.exists():
        g = block_geo(block_id)
        it = _items(int(day[:4]), g["tile"])
        d = pd.to_datetime(it["acq_dt"], utc=True).dt.strftime("%Y-%m-%d")
        sel = it[(d == day) & (it["sensor"] == sensor)]
        if sel.empty:
            raise LookupError(f"no {sensor} granule for {block_id} on {day}")
        row = sel.iloc[0]
        r = g["row"]
        from rasterio.windows import Window

        win = Window(int(r["col_off"]), int(r["row_off"]), int(r["width"]), int(r["height"]))
        hrefs = dict(blue=row["href_blue"], red=row["href_red"], nir=row["href_nir"],
                     fmask=row["href_fmask"], green=row["href_red"].replace(".B04.", ".B03."))
        arrs = {k: hls.read_windows(h, [win])[0] for k, h in hrefs.items()}
        np.savez_compressed(path, granule=row["id"], **arrs)
    z = np.load(path, allow_pickle=False)
    blue, green, red, nir = (hls.scale_reflectance(z[k]) for k in ("blue", "green", "red", "nir"))
    ndvi, _, ok = hls.vegetation_indices(blue, red, nir)
    clear = hls.fmask_clear(z["fmask"]) & ok
    return dict(rgb_raw=np.dstack([red, green, blue]), ndvi=np.where(clear, ndvi, np.nan),
                clear=clear, granule=str(z["granule"]))


def true_colour(rgb: np.ndarray, lo_hi=None, gamma: float = 0.85) -> np.ndarray:
    """Percentile stretch (pooled over bands, keeps colour balance) + slight gamma."""
    v = np.nan_to_num(rgb, nan=0.0)
    if lo_hi is None:
        lo_hi = (np.percentile(v[v > 0], 2), np.percentile(v[v > 0], 98))
    lo, hi = lo_hi
    out = np.clip((v - lo) / (hi - lo), 0, 1) ** gamma
    return out


def modis_ndvi(block_id: str, day: str) -> dict:
    """MOD13Q1 (Terra, v061) NDVI of the 16-day composite holding ``day``, nearest-sampled
    onto the block's 30 m grid so each 250 m pixel shows as a block (cached)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"modis240_{block_id}_{day}.npz"
    if not path.exists():
        import planetary_computer as pc
        import pystac_client
        import rasterio
        from affine import Affine
        from rasterio.vrt import WarpedVRT
        from rasterio.warp import Resampling

        g = block_geo(block_id)
        x0, y0, x1, y1 = g["bounds"]
        t = Affine(240.0, 0, x0, 0, -240.0, y1)  # 32 x 32 cells ~ native 250 m pixels
        d0 = date.fromisoformat(day)
        cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
        bb = (g["lon"] - 0.05, g["lat"] - 0.05, g["lon"] + 0.05, g["lat"] + 0.05)
        items = list(cat.search(collections=["modis-13Q1-061"], bbox=bb,
                                datetime=f"{d0 - timedelta(days=20)}/{d0 + timedelta(days=20)}"
                                ).items())
        best = None
        for it in items:
            if not it.id.startswith("MOD13Q1"):
                continue
            s = date.fromisoformat(it.properties["start_datetime"][:10])
            e = date.fromisoformat(it.properties["end_datetime"][:10])
            if s <= d0 <= e:
                best = (it, s, e)
        if best is None:
            raise LookupError(f"no Terra composite for {day}")
        it, s, e = best

        def read(asset):
            href = pc.sign(it.assets[asset].href)
            with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"), rasterio.open(href) as src:
                with WarpedVRT(src, crs=f"EPSG:{g['epsg']}", transform=t, width=32, height=32,
                               resampling=Resampling.nearest, nodata=src.nodata) as vrt:
                    return vrt.read(1)

        nd, rel = read("250m_16_days_NDVI"), read("250m_16_days_pixel_reliability")
        np.savez_compressed(path, ndvi=nd, rel=rel, start=str(s), end=str(e), id=it.id)
    z = np.load(path, allow_pickle=False)
    nd = z["ndvi"].astype("float32") * 1e-4
    ok = (z["ndvi"] != -3000) & np.isin(z["rel"], (0, 1))
    up = lambda a: np.kron(a, np.ones((8, 8), dtype=a.dtype))  # noqa: E731  32 -> 256 px
    return dict(ndvi=np.where(up(ok), up(nd), np.nan), start=str(z["start"]), end=str(z["end"]))


# =====================================================================================
# plotting helpers
# =====================================================================================
def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "savefig.facecolor": "white",
    })


def _clean(ax, grey_bg=True):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if grey_bg:
        ax.set_facecolor("#e6e6e6")


def _scalebar(ax, km=2.0, color="white"):
    px = km * 1000 / 30.0
    x, y = 12, 256 - 14
    ax.plot([x, x + px], [y, y], color=color, lw=2.5, solid_capstyle="butt")
    ax.text(x + px / 2, y - 5, f"{km:g} km", color=color, ha="center", va="bottom", fontsize=8,
            fontweight="bold")


def _outline(ax, mask, color="#ffffff", lw=0.6):
    ax.contour(mask.astype(float), levels=[0.5], colors=["#101010"], linewidths=lw * 2.6)
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _save(fig, name, max_kb=400, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    dpi = kw.pop("dpi", PNG_DPI)
    while True:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, **kw)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        q = im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        out = io.BytesIO()
        q.save(out, format="PNG", optimize=True)
        if out.tell() / 1024 <= max_kb or dpi <= 60:
            break
        dpi = int(dpi * 0.85)
    path.write_bytes(out.getvalue())
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB  ({im.size[0]}x{im.size[1]})")


def _colorbar(fig, ax, label, **kw):
    sm = plt.cm.ScalarMappable(cmap=NDVI_KW["cmap"],
                               norm=plt.Normalize(NDVI_KW["vmin"], NDVI_KW["vmax"]))
    cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.03, **kw)
    cb.set_label(label, fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    return cb


def _mask(block_id: str, year: int) -> np.ndarray:
    return hlssample.season_mask(block_id, year)


# =====================================================================================
# 1. hero
# =====================================================================================
def make_hero():
    h = HERO
    g = block_geo(h["block"])
    sc = hls_scene(h["block"], h["day"], h["sensor"])
    name = {"S30": "Sentinel-2", "L30": "Landsat 8/9"}[h["sensor"]]
    W, H = 13.0, 5.7
    fig = plt.figure(figsize=(W, H))
    ps = 4.0  # panel side, inches
    y0 = 0.10 * H
    a1 = fig.add_axes([0.02, y0 / H, ps / W, ps / H])
    a2 = fig.add_axes([0.02 + ps / W + 0.012, y0 / H, ps / W, ps / H])
    cax = fig.add_axes([0.02 + 2 * ps / W + 0.03, y0 / H, 0.010, ps / H])
    a1.imshow(true_colour(sc["rgb_raw"]), interpolation="nearest")
    a2.imshow(sc["ndvi"], interpolation="nearest", **NDVI_KW)
    for a in (a1, a2):
        _clean(a)
    _scalebar(a1)
    a1.set_title("True colour", fontsize=11, loc="left", pad=6)
    a2.set_title("NDVI (plant greenness)", fontsize=11, loc="left", pad=6)
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=NDVI_KW["cmap"], norm=plt.Normalize(0.3, 0.95)),
                      cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    fig.text(0.02, 0.935, "CornSat", fontsize=28, fontweight="bold", color="#2e6b2e", va="center")
    xt = 0.02 + 2 * ps / W + 0.085
    fig.text(xt, 0.78, "Can satellites see the\nUS corn crop before\nthe market does?", fontsize=15,
             va="top", linespacing=1.35)
    fig.text(xt, 0.44, "NASA HLS 30 m\n"
             f"{name} ({h['sensor']}), {h['day']}\ncentral Iowa, {_ll(g['lat'], g['lon'])}\n"
             "one sample block, 7.68 x 7.68 km\n256 x 256 pixels; grey = cloud or shadow",
             fontsize=10.5, color="#444444", va="top", linespacing=1.6)
    _save(fig, "hero.png")


# =====================================================================================
# 2. resolution
# =====================================================================================
def make_resolution():
    h = HERO
    g = block_geo(h["block"])
    year = int(h["day"][:4])
    m = _mask(h["block"], year)
    sc = hls_scene(h["block"], h["day"], h["sensor"])
    mo = modis_ndvi(h["block"], h["day"])
    fig = plt.figure(figsize=(11, 5.4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.045], wspace=0.05, left=0.02, right=0.93,
                           top=0.84, bottom=0.12)
    a1, a2, cax = (fig.add_subplot(gs[0, i]) for i in range(3))
    for a, arr in ((a1, mo["ndvi"]), (a2, sc["ndvi"])):
        a.imshow(arr, interpolation="nearest", **NDVI_KW)
        _outline(a, m)
        _clean(a)
        _scalebar(a)
    sel = m & np.isfinite(sc["ndvi"]) & np.isfinite(mo["ndvi"])
    hm, mm = np.nanmean(sc["ndvi"][sel]), np.nanmean(mo["ndvi"][sel])
    a1.set_title("MODIS 250 m (Terra MOD13Q1)", fontsize=11, loc="left", fontweight="bold")
    a2.set_title("HLS 30 m (Sentinel-2)", fontsize=11, loc="left", fontweight="bold")
    a1.text(0.02, 0.97, f"{mo['start']} to {mo['end']} composite\nmean NDVI on corn {mm:.2f}",
            transform=a1.transAxes, va="top", fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=.8))
    a2.text(0.02, 0.97, f"{h['day']}\nmean NDVI on corn {hm:.2f}", transform=a2.transAxes, va="top",
            fontsize=8.5, bbox=dict(fc="white", ec="none", alpha=.8))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=NDVI_KW["cmap"], norm=plt.Normalize(0.3, 0.95)),
                      cax=cax)
    cb.set_label("NDVI", fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    fig.suptitle("Same 7.7 km block, two resolutions: outlined = predicted corn fields",
                 x=0.02, y=0.97, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.04,
             f"Central Iowa, {_ll(g['lat'], g['lon'])}. A MODIS pixel mixes corn, soy and roads; "
             "a 30 m pixel mostly does not.\nThe corn mask is predicted from CDL of earlier years "
             "(soy-after-corn rotation), no look-ahead.", fontsize=8.8, color="#444444")
    _save(fig, "resolution.png")


# =====================================================================================
# 3. sampling map
# =====================================================================================
def make_sampling_map():
    from cornsat.aoi import _states

    allst = _states().to_crs(5070)
    gdf = allst[allst["STUSPS"].isin(list(STATES))].copy()
    bl = hlssample.load_blocks().copy()
    pts = []
    for r in bl.itertuples():
        x0, y0, x1, y1 = r.bounds_utm
        pts.append(Transformer.from_crs(int(r.epsg), 5070, always_xy=True).transform(
            (x0 + x1) / 2, (y0 + y1) / 2))
    bl["x"], bl["y"] = zip(*pts)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    allst.plot(ax=ax, color="#f1f1ee", edgecolor="#cfcfc8", lw=0.5)
    gdf.plot(ax=ax, color="#fbfaf3", edgecolor="#555555", lw=1.0)
    sc = ax.scatter(bl["x"], bl["y"], c=bl["corn_frac_0812"], cmap="YlOrBr", vmin=0.2, vmax=0.8,
                    s=34, edgecolor="#333333", linewidth=0.4, zorder=3)
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 90_000
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    for _, r in gdf.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(r["STUSPS"], (c.x, c.y), ha="center", va="center", fontsize=15,
                    color="#00000033", fontweight="bold", zorder=1)
    ax.set_axis_off()
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("CDL corn share of block, 2008-2012 mean", fontsize=9)
    cb.ax.yaxis.set_label_position("left")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    ax.set_title(f"{len(bl)} fixed HLS sample blocks (7.7 km each), 72 per state",
                 loc="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.025,
             "Blocks are picked once, using CDL 2008-2012 only, so the choice cannot leak any "
             "modelled season. Equal-area projection (EPSG:5070).", fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.01, right=0.97, top=0.93, bottom=0.07)
    _save(fig, "sampling_map.png")


# =====================================================================================
# 4. derecho
# =====================================================================================
def make_derecho():
    d = DERECHO
    g = block_geo(d["block"])
    b = hls_scene(d["block"], *d["before"])
    a = hls_scene(d["block"], *d["after"])
    m = _mask(d["block"], 2020)
    # same stretch for both dates so the colour change is real
    both = np.concatenate([np.nan_to_num(b["rgb_raw"]).ravel(), np.nan_to_num(a["rgb_raw"]).ravel()])
    both = both[both > 0]
    lo_hi = (np.percentile(both, 2), np.percentile(both, 98))
    dn = a["ndvi"] - b["ndvi"]
    nb = np.nanmean(np.where(m, b["ndvi"], np.nan))
    na = np.nanmean(np.where(m, a["ndvi"], np.nan))
    fig = plt.figure(figsize=(11.5, 6.8))
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 0.04], hspace=0.12, wspace=0.05,
                           left=0.02, right=0.95, top=0.86, bottom=0.09)
    ax = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    cax = fig.add_subplot(gs[:, 3])
    ax[0][0].imshow(true_colour(b["rgb_raw"], lo_hi), interpolation="nearest")
    ax[0][1].imshow(true_colour(a["rgb_raw"], lo_hi), interpolation="nearest")
    ax[1][0].imshow(b["ndvi"], interpolation="nearest", **NDVI_KW)
    ax[1][1].imshow(a["ndvi"], interpolation="nearest", **NDVI_KW)
    im = ax[0][2].imshow(dn, cmap="RdBu", vmin=-0.4, vmax=0.4, interpolation="nearest")
    ax[1][2].imshow(np.where(m, dn, np.nan), cmap="RdBu", vmin=-0.4, vmax=0.4, interpolation="nearest")
    for r in ax:
        for x in r:
            _clean(x)
    _scalebar(ax[0][0])
    ax[0][0].set_title(f"Before: {d['before'][0]}", loc="left", fontsize=11, fontweight="bold")
    ax[0][1].set_title(f"After: {d['after'][0]}", loc="left", fontsize=11, fontweight="bold")
    ax[0][2].set_title("NDVI change, all pixels", loc="left", fontsize=11, fontweight="bold")
    ax[1][2].set_title("NDVI change, predicted corn only", loc="left", fontsize=10, pad=4)
    ax[0][0].set_ylabel("True colour", fontsize=10)
    ax[1][0].set_ylabel("NDVI", fontsize=10)
    for x in (ax[0][0], ax[1][0]):
        x.yaxis.set_label_position("left")
        x.set_ylabel(x.get_ylabel(), fontsize=10, labelpad=4)
    ax[1][0].text(0.02, 0.97, f"corn NDVI {nb:.3f}", transform=ax[1][0].transAxes, va="top", fontsize=9,
                  bbox=dict(fc="white", ec="none", alpha=.8))
    ax[1][1].text(0.02, 0.97, f"corn NDVI {na:.3f}", transform=ax[1][1].transAxes, va="top", fontsize=9,
                  bbox=dict(fc="white", ec="none", alpha=.8))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("NDVI change (after - before)", fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    fig.suptitle("Iowa derecho, 10 August 2020, seen at 30 m", x=0.02, y=0.975, ha="left",
                 fontsize=14, fontweight="bold")
    fig.text(0.02, 0.915, "Corn flattened by winds above 100 mph loses greenness within days.",
             fontsize=10.5, color="#444444")
    fig.text(0.02, 0.03,
             f"NASA HLS 30 m, Sentinel-2. Sample block about 40 km north of Cedar Rapids, Iowa, "
             f"{_ll(g['lat'], g['lon'])}. NDVI colours as in the other figures; same colour stretch "
             "on both dates.", fontsize=8.8, color="#444444")
    _save(fig, "derecho_2020.png")


# =====================================================================================
# 5. season gif
# =====================================================================================
def season_frames(block_id: str, year: int, min_gap: int = 6, min_clear: float = 0.85):
    s = pd.read_parquet(f"data/interim/hls/scenes/{year}/{block_geo(block_id)['tile']}.parquet")
    s = s[(s["block_id"] == block_id)].copy()
    s["day"] = pd.to_datetime(s["acq_dt"], utc=True).dt.tz_convert(None).dt.normalize()
    s["cf"] = s["n_clear"] / s["n_mask"]
    curve = s[s["ndvi_mean"].notna()].groupby("day")["ndvi_mean"].mean()
    ok = s[s["cf"] >= min_clear].sort_values(["day", "cf"], ascending=[True, False])
    ok = ok.drop_duplicates("day")
    keep, last = [], None
    for _, r in ok.iterrows():
        if last is None or (r["day"] - last).days >= min_gap:
            keep.append((r["day"].strftime("%Y-%m-%d"), r["sensor"]))
            last = r["day"]
    return keep, curve


def make_season(width_px=760):
    b, year = SEASON["block"], SEASON["year"]
    g = block_geo(b)
    m = _mask(b, year)
    frames, curve = season_frames(b, year)
    # one fixed stretch for the whole season from the clearest summer frame
    ref = hls_scene(b, *max(frames, key=lambda f: -abs((date.fromisoformat(f[0]) - date(year, 7, 20)).days)))
    v = np.nan_to_num(ref["rgb_raw"])
    lo_hi = (np.percentile(v[v > 0], 1), np.percentile(v[v > 0], 99.5))
    imgs = []
    for day, sensor in frames:
        sc = hls_scene(b, day, sensor)
        W, H, ps = 7.6, 4.4, 2.5
        fig = plt.figure(figsize=(W, H), dpi=100)
        a1 = fig.add_axes([0.02, 0.30, ps / W, ps / H])
        a2 = fig.add_axes([0.02 + ps / W + 0.012, 0.30, ps / W, ps / H])
        a3 = fig.add_axes([0.09, 0.09, 0.88, 0.14])
        a1.imshow(true_colour(sc["rgb_raw"], lo_hi), interpolation="nearest")
        a2.imshow(sc["ndvi"], interpolation="nearest", **{**NDVI_KW, "vmin": 0.1})
        _clean(a1)
        _clean(a2)
        a1.set_title("True colour", fontsize=9.5, loc="left", pad=3)
        a2.set_title("NDVI", fontsize=9.5, loc="left", pad=3)
        sensor_name = {"S30": "Sentinel-2", "L30": "Landsat 8/9"}
        a3.plot(curve.index, curve.values, color="#9fc79f", lw=1.2)
        cur = pd.Timestamp(day)
        a3.plot(curve.index[curve.index <= cur], curve.values[curve.index <= cur], color="#2e6b2e",
                lw=2)
        a3.scatter([cur], [curve.get(cur, np.nan)], color="#d9531e", zorder=3, s=30)
        a3.set_xlim(pd.Timestamp(year, 4, 28), pd.Timestamp(year, 10, 2))
        a3.set_ylim(0.1, 1.0)
        a3.set_yticks([0.2, 0.6, 0.9])
        a3.set_xticks([pd.Timestamp(year, mth, 1) for mth in range(5, 11)])
        a3.set_xticklabels(["May", "Jun", "Jul", "Aug", "Sep", "Oct"], fontsize=8)
        a3.tick_params(labelsize=8, length=2)
        a3.set_ylabel("corn NDVI", fontsize=8)
        for s in ("top", "right"):
            a3.spines[s].set_visible(False)
        xr = 0.02 + 2 * ps / W + 0.04
        fig.text(xr, 0.80, f"{day}", fontsize=18, fontweight="bold", color="#2e6b2e", va="center")
        fig.text(xr, 0.66, f"NASA HLS 30 m\n{sensor_name[sensor]}\none Iowa block\n"
                 f"{_ll(g['lat'], g['lon'])}\n7.7 x 7.7 km", fontsize=9, va="top",
                 color="#444444", linespacing=1.5)
        fig.text(xr, 0.33, "grey = cloud or shadow\nNDVI: pale = bare soil,\ndark green = full canopy", fontsize=8,
                 va="top", color="#666666", linespacing=1.5)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        im = im.resize((width_px, round(im.size[1] * width_px / im.size[0])), Image.LANCZOS)
        imgs.append(im)
    # shared palette for stable colours, then GIF
    pal = Image.new("RGB", (width_px, imgs[0].size[1] * len(imgs)))
    for i, im in enumerate(imgs):
        pal.paste(im, (0, i * im.size[1]))
    pal = pal.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "season.gif"
    q[0].save(path, save_all=True, append_images=q[1:], duration=[900] * (len(q) - 1) + [2200],
              loop=0, optimize=True, disposal=1)
    print(f"wrote {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.0f} KB  "
          f"({len(q)} frames: {', '.join(f[0][5:] for f in frames)})")


# =====================================================================================
# 6. results
# =====================================================================================
def make_results():
    d = pd.read_csv(ROOT / "reports" / "state_panel_accuracy.csv")
    d = d[(d["window"] == "2016+") & (d["scheme"] == "loso") & (d["snapshot"] == "ALL")]
    rm = d.drop_duplicates("set").set_index("set")["rmse"]
    order = [("trend", "trend\nonly"), ("W+C", "weather +\ncrop cond."),
             ("W+C+S", "+ MODIS\n(250 m)"), ("W+C+H", "+ HLS\n(30 m)")]
    cols = [GREY, "#4c6a8a", "#2e7d5b", "#d9822b"]
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    vals = [rm[k] for k, _ in order]
    bars = ax.bar([lab for _, lab in order], vals, color=cols, width=0.62)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")
    ax.set_ylim(9.5, 12.4)
    ax.set_ylabel("RMSE of yield surprise, bu/ac (lower is better)", fontsize=9)
    ax.tick_params(length=0, labelsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color="#e3e3e3", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Satellites add little to weather + USDA crop condition", loc="left", fontsize=12,
                 fontweight="bold")
    fig.text(0.01, 0.015, "5-state panel, 2016+, leave-one-season-out, all 6 snapshots. "
             "Bar axis starts at 9.5. Gain is small and fragile (gone with a 14-day lag).",
             fontsize=8, color="#444444")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.2)
    _save(fig, "results.png")


STEPS = dict(hero=make_hero, resolution=make_resolution, sampling_map=make_sampling_map,
             derecho=make_derecho, season=make_season, results=make_results)


def main():
    global CACHE
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=list(STEPS), action="append", help="build only this image")
    ap.add_argument("--cache", type=Path, help="download cache dir")
    args = ap.parse_args()
    if args.cache:
        CACHE = args.cache
    _style()
    for name in args.only or STEPS:
        print(f"== {name}")
        STEPS[name]()


if __name__ == "__main__":
    main()

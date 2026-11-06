import itertools

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import box

from cornsat import cornmask, hlssample
from cornsat.hlssample import BLOCK, CORE_PX, CORN, MARGIN, SOY

OTHER = 36  # alfalfa


# --------------------------------------------------------------------------------------
# rotation rule and erosion
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("p1,p2,p3", list(itertools.product([CORN, SOY, OTHER, 0], repeat=3)))
def test_rotation_rule_all_cases(p1, p2, p3):
    a = lambda v: np.full((2, 2), v, dtype="uint8")  # noqa: E731
    got = hlssample.rotation_corn(a(p1), a(p2), a(p3))
    expect = (p1 == SOY and p2 == CORN) or (p1 == CORN and p2 == CORN and p3 == CORN)
    assert got.dtype == bool
    assert got.all() == expect and got.any() == expect


@pytest.mark.parametrize("p1,p2", list(itertools.product([CORN, SOY, OTHER], repeat=2)))
def test_rotation_rule_without_third_year(p1, p2):
    a = lambda v: np.full((1, 1), v, dtype="uint8")  # noqa: E731
    got = bool(hlssample.rotation_corn(a(p1), a(p2))[0, 0])
    assert got == ((p1 == SOY and p2 == CORN) or (p1 == CORN and p2 == CORN))


def test_rotation_rule_pixelwise_mixture():
    y1 = np.array([[SOY, CORN, CORN, SOY]], dtype="uint8")
    y2 = np.array([[CORN, CORN, CORN, SOY]], dtype="uint8")
    y3 = np.array([[OTHER, CORN, SOY, CORN]], dtype="uint8")
    assert hlssample.rotation_corn(y1, y2, y3).tolist() == [[True, True, False, False]]


def test_erosion_removes_edge_pixels():
    m = np.zeros((11, 11), bool)
    m[2:9, 2:9] = True  # 7x7 field
    m[0, :] = True  # one-pixel-wide strip
    e = hlssample.erode(m)
    assert e[3:8, 3:8].all() and e.sum() == 25  # 5x5 interior survives
    assert not e[0].any()
    assert not e[2, 2:9].any() and not e[2:9, 2].any()  # field edge removed


def test_erosion_treats_outside_as_false():
    e = hlssample.erode(np.ones((5, 5), bool))
    assert e.sum() == 9 and e[1:4, 1:4].all()


# --------------------------------------------------------------------------------------
# no look-ahead in the season mask
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("season", [2010, 2013, 2015, 2020, 2024, 2026])
def test_season_cdl_years(season):
    ys = hlssample.season_cdl_years(season)
    assert ys[0] == season - 1 and ys[1] == season - 2
    assert all(y < season for y in ys)
    assert all(cornmask.CDL_FIRST_YEAR <= y <= cornmask.CDL_LAST_YEAR for y in ys)
    assert len(ys) == (2 if season == 2010 else 3)


def test_season_cdl_years_out_of_range():
    with pytest.raises(ValueError):
        hlssample.season_cdl_years(cornmask.CDL_LAST_YEAR + 2)
    with pytest.raises(ValueError):
        hlssample.season_cdl_years(cornmask.CDL_FIRST_YEAR + 1)


def _fake_cdl(year: int) -> np.ndarray:
    """Synthetic block (+margin) where the left half rotates corn/soy, the right half is
    continuous corn until 2022 then alfalfa."""
    n = BLOCK + 2 * MARGIN
    a = np.full((n, n), OTHER, dtype="uint8")
    a[:, : n // 2] = CORN if year % 2 == 0 else SOY
    a[:, n // 2 :] = CORN if year < 2022 else OTHER
    return a


@pytest.fixture
def recording_cdl(monkeypatch, tmp_path):
    calls = []

    def fake_block_cdl(state, year, refresh=False):
        calls.append((state, int(year)))
        return {f"{state}_T15TXX_r0000_c0000": _fake_cdl(year), f"{state}_T15TXX_r0256_c0000": _fake_cdl(year)}

    monkeypatch.setattr(hlssample, "HLS_DIR", tmp_path)
    monkeypatch.setattr(hlssample, "block_cdl", fake_block_cdl)
    hlssample._load_masks.cache_clear()
    yield calls
    hlssample._load_masks.cache_clear()


@pytest.mark.parametrize("season", [2013, 2016, 2021, 2024, 2026])
def test_mask_uses_only_prior_cdl_years(recording_cdl, season):
    hlssample.build_masks(season, ["IA"])
    years = [y for _, y in recording_cdl]
    assert years and all(y < season for y in years)
    assert sorted(set(years)) == sorted({season - 1, season - 2, season - 3})
    assert hlssample.mask_meta("IA", season)["cdl_years"] == [season - 1, season - 2, season - 3]
    m = hlssample.season_mask("IA_T15TXX_r0000_c0000", season)
    assert m.shape == (BLOCK, BLOCK) and m.dtype == bool
    left, right = m[1:-1, 1 : BLOCK // 2 - 2], m[1:-1, BLOCK // 2 + 2 : -1]
    # left half: CDL Y-1 soy & Y-2 corn  <=> Y-1 odd
    assert left.all() == ((season - 1) % 2 == 1) and left.any() == left.all()
    # right half: continuous corn through Y-1 only if Y-1 < 2022
    assert right.all() == (season - 1 < 2022) and right.any() == right.all()


def test_mask_purity_uses_season_cdl_only_as_diagnostic(recording_cdl):
    hlssample.build_masks(2021, ["IA"])
    recording_cdl.clear()
    p = hlssample.mask_purity("IA", 2021)
    assert {y for _, y in recording_cdl} == {2021}
    assert 0.0 <= p <= 1.0
    with pytest.raises(ValueError):
        hlssample.mask_purity("IA", cornmask.CDL_LAST_YEAR + 1)


def test_mask_cache_is_packed_and_reloaded(recording_cdl, tmp_path):
    paths = hlssample.build_masks(2020, ["IA"])
    assert paths["IA"] == tmp_path / "masks" / "IA_2020.npz" and paths["IA"].exists()
    n = len(recording_cdl)
    hlssample.build_masks(2020, ["IA"])  # cached: no new CDL reads
    assert len(recording_cdl) == n
    with np.load(paths["IA"]) as z:
        assert z["IA_T15TXX_r0000_c0000"].dtype == np.uint8
        assert z["IA_T15TXX_r0000_c0000"].size == BLOCK * BLOCK // 8


# --------------------------------------------------------------------------------------
# candidate blocks / selection geometry (synthetic tiles across the UTM 14/15 seam)
# --------------------------------------------------------------------------------------
TILES = [
    {"tile": "T15A", "epsg": 32615, "ulx": 199980.0, "uly": 4700040.0},
    {"tile": "T15B", "epsg": 32615, "ulx": 299980.0, "uly": 4700040.0},
    {"tile": "T15C", "epsg": 32615, "ulx": 199980.0, "uly": 4600040.0},
    {"tile": "T14A", "epsg": 32614, "ulx": 699960.0, "uly": 4700040.0},  # overlaps T15A
]


@pytest.fixture(scope="module")
def synthetic_selection():
    state = hlssample._reproject_geom(box(-97.6, 40.9, -94.4, 42.6), "EPSG:4326", "EPSG:5070", seg=0.05)
    inset = state.buffer(-hlssample.STATE_INSET_M)
    cands = pd.concat(
        [hlssample.candidate_blocks(t, "XX", inset_geom=inset) for t in TILES], ignore_index=True
    )
    # corn concentrated near the -96 deg meridian (zone seam) so both zones' best blocks
    # compete for the same ground
    seam_x = hlssample._reproject_geom(shapely.Point(-96.0, 41.7), "EPSG:4326", "EPSG:5070").x
    cx = np.array([g.centroid.x for g in cands["geometry"]])
    cands["score"] = 1.5 + np.exp(-(((cx - seam_x) / 30_000.0) ** 2))  # all tiles eligible
    sel = hlssample.select_blocks(cands, tiles_per_state=4, blocks_per_tile=9, spacings=(0.0,))
    return state, inset, cands, sel


def test_candidates_aligned_and_in_core(synthetic_selection):
    _, inset, cands, _ = synthetic_selection
    assert len(cands) > 100
    for col in ("row_off", "col_off"):
        assert (cands[col] % BLOCK == 0).all()
        assert (cands[col] >= 0).all() and (cands[col] + BLOCK <= CORE_PX).all()
    assert shapely.contains(inset, cands["geometry"].values).all()


def test_selected_blocks_do_not_overlap(synthetic_selection):
    _, _, cands, sel = synthetic_selection
    geoms = list(sel["geometry"])
    assert len(geoms) == 4 * 9 and set(sel["tile"]) == {t["tile"] for t in TILES}
    for a, b in itertools.combinations(geoms, 2):
        assert a.intersection(b).area < 1.0
    # the test is meaningful: without the overlap rule the two zones' top blocks collide
    top = lambda t: cands[cands["tile"] == t].nlargest(9, "score")["geometry"]  # noqa: E731
    assert any(a.intersection(b).area > 1.0 for a in top("T15A") for b in top("T14A"))


def test_selected_blocks_inside_state_and_aligned(synthetic_selection):
    state, _, _, sel = synthetic_selection
    assert shapely.contains(state, sel["geometry"].values).all()
    assert (sel["row_off"] % BLOCK == 0).all() and (sel["col_off"] % BLOCK == 0).all()
    assert (sel["row_off"] + BLOCK <= CORE_PX).all() and (sel["col_off"] + BLOCK <= CORE_PX).all()
    assert sel["rank"].tolist() == list(range(1, len(sel) + 1))


def test_selection_is_deterministic(synthetic_selection):
    _, _, cands, sel = synthetic_selection
    again = hlssample.select_blocks(
        cands.sample(frac=1.0, random_state=1), tiles_per_state=4, blocks_per_tile=9, spacings=(0.0,)
    )
    assert again["block_id"].tolist() == sel["block_id"].tolist()


def test_spacing_never_forces_low_corn_tiles(synthetic_selection):
    _, _, cands, _ = synthetic_selection
    c = cands.copy()
    c["score"] = c["score"] - 1.5  # T15B (far from the seam) now has little corn
    sel = hlssample.select_blocks(c, tiles_per_state=4, blocks_per_tile=9, spacings=(1e9, 0.0))
    assert "T15B" not in set(sel["tile"])
    assert sel["tile"].iloc[0] in {"T14A", "T15A"}  # highest corn mass first


def test_score_candidates_on_synthetic_weight():
    t = TILES[1]
    cands = hlssample.candidate_blocks(t, "XX", inset_geom=box(-1e7, -1e7, 1e7, 1e7))
    # weight raster in UTM 15 at 240 m: 1.0 on the western half of the tile core, else 0
    from affine import Affine

    aff = Affine(240.0, 0, t["ulx"], 0, -240.0, t["uly"])
    w = np.zeros((420, 420))
    w[:, : (CORE_PX * 30 // 2) // 240] = 1.0
    s = hlssample.score_candidates(cands, w, aff, "EPSG:32615")
    west = cands["col_off"] + BLOCK <= (CORE_PX * 30 // 2) // 30 - 8
    east = cands["col_off"] >= (CORE_PX * 30 // 2) // 30 + 8
    assert np.allclose(s[west.values], 1.0) and np.allclose(s[east.values], 0.0)


def test_block_selection_uses_only_cdl_2008_2012(monkeypatch):
    try:
        gb = cornmask.modis_geobox("IA")
    except Exception as e:  # boundaries unavailable
        pytest.skip(f"boundaries unavailable: {e}")
    seen = []

    def fake_corn_fraction(state, year, refresh=False):
        seen.append(int(year))
        return cornmask._to_dataarray(np.full(gb.shape, 0.1, "float32"), gb, "corn_fraction")

    monkeypatch.setattr(cornmask, "corn_fraction", fake_corn_fraction)
    w, aff, crs = hlssample._selection_weight("IA")
    assert sorted(seen) == [2008, 2009, 2010, 2011, 2012]
    assert max(seen) < 2013  # pre-dates every HLS season incl. the 2013-2015 presample
    assert np.allclose(w, 0.1)


# --------------------------------------------------------------------------------------
# the real cached block table (skipped when it has not been built)
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def real_blocks():
    path = hlssample._blocks_path()
    if not path.exists():
        pytest.skip("blocks.parquet not built")
    return pd.read_parquet(path)


def test_real_blocks_columns_and_alignment(real_blocks):
    b = real_blocks
    for c in ("block_id", "state", "tile", "epsg", "row_off", "col_off", "height", "width",
              "transform", "bounds_utm", "geometry_5070", "corn_frac_0812", "rank"):
        assert c in b.columns
    assert b["block_id"].is_unique
    assert (b["row_off"] % BLOCK == 0).all() and (b["col_off"] % BLOCK == 0).all()
    assert (b[["row_off", "col_off"]] + BLOCK <= CORE_PX).all().all()
    assert b["block_id"].str.split("_").str[0].eq(b["state"]).all()


def test_real_blocks_disjoint_and_inside_states(real_blocks):
    geoms = shapely.from_wkb(real_blocks["geometry_5070"].values)
    tree = shapely.STRtree(geoms)
    for i, g in enumerate(geoms):
        for j in tree.query(g):
            if j != i:
                assert g.intersection(geoms[j]).area < 1.0
    for st, idx in real_blocks.groupby("state").groups.items():
        assert shapely.contains(hlssample._state_geom_5070(st), geoms[idx]).all()


# --------------------------------------------------------------------------------------
# network (skip cleanly when offline)
# --------------------------------------------------------------------------------------
def test_hls_cog_block_size():
    try:
        import planetary_computer as pc
        import pystac_client
        import rasterio

        cat = pystac_client.Client.open(hlssample.STAC_URL, modifier=pc.sign_inplace)
        it = next(
            cat.search(collections=["hls2-s30"], bbox=[-93.6, 41.9, -93.5, 42.0],
                       datetime="2023-07-01/2023-07-31", limit=1).items()
        )
        with rasterio.open(it.assets["B04"].href) as ds:
            shapes, size = ds.block_shapes, (ds.height, ds.width)
    except Exception as e:
        pytest.skip(f"Planetary Computer unavailable: {e}")
    assert shapes == [(BLOCK, BLOCK)]
    assert size == (hlssample.TILE_PX, hlssample.TILE_PX)


@pytest.mark.parametrize("year", [2012, 2023])
def test_cdl_codes_window(year):
    b = (200_000.0, 2_100_000.0, 207_680.0, 2_107_680.0)  # central Iowa, EPSG:5070
    try:
        codes, aff = cornmask.cdl_codes_window(year, b, "IA")
    except Exception as e:
        pytest.skip(f"CDL unavailable: {e}")
    assert codes.dtype == np.uint8
    assert ((aff.c - cornmask.CDL_GRID_OFFSET) / 30) % 1 == 0
    assert aff.c <= b[0] and aff.f >= b[3]
    assert codes.shape[0] * 30 >= b[3] - b[1] and codes.shape[1] * 30 >= b[2] - b[0]
    assert (codes == CORN).mean() > 0.2 and (codes == SOY).mean() > 0.1


def test_block_cdl_matches_modis_corn_fraction(real_blocks):
    """Geolocation check: corn share of the resampled 30 m CDL in a block ~ the cached
    MODIS-grid corn fraction averaged over the same block."""
    r = real_blocks[real_blocks["state"] == "IA"].iloc[0]
    try:
        codes = hlssample._block_codes(r, 2021)
        frac = cornmask.corn_fraction("IA", 2021)
    except Exception as e:
        pytest.skip(f"CDL unavailable: {e}")
    cand = pd.DataFrame([{**r.to_dict(), "ulx": r["transform"][2] - r["col_off"] * 30,
                          "uly": r["transform"][5] + r["row_off"] * 30}])
    s = hlssample.score_candidates(cand, frac.values.astype("float64"),
                                   cornmask.modis_geobox("IA").affine, cornmask.MODIS_PROJ)[0]
    share = (codes[MARGIN:-MARGIN, MARGIN:-MARGIN] == CORN).mean()
    assert abs(share - s) < 0.03

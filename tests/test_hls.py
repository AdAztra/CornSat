"""Phase 8c: HLS ingestion and causal composites (synthetic; network tests skip)."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from cornsat import features, hls


# --- Fmask policy -----------------------------------------------------------------------
@pytest.mark.parametrize("bit", [0, 1, 2, 3, 4, 5])
def test_fmask_each_rejected_bit(bit):
    assert not hls.fmask_clear(np.array([1 << bit], np.uint8))[0]


def test_fmask_aerosol_levels_and_nodata():
    vals = np.array([0, 0b01 << 6, 0b10 << 6, 0b11 << 6, 255], np.uint8)
    assert hls.fmask_clear(vals).tolist() == [True, True, True, False, False]
    # high aerosol alone rejects; moderate aerosol + water rejects through bit 5
    assert not hls.fmask_clear(np.array([(0b10 << 6) | (1 << 5)], np.uint8))[0]


def test_fmask_all_bytes_match_policy():
    v = np.arange(256, dtype=np.uint8)
    expect = [(x != 255) and (x & 0x3F == 0) and ((x >> 6) != 3) for x in range(256)]
    assert hls.fmask_clear(v).tolist() == expect
    assert hls.fmask_clear(v).sum() == 3  # 0, 64, 128


# --- bands, scale, indices -----------------------------------------------------------------
def test_band_mapping_per_sensor():
    assert hls.band_names("L30") == {"blue": "B02", "red": "B04", "nir": "B05"}
    assert hls.band_names("S30") == {"blue": "B02", "red": "B04", "nir": "B8A"}


def test_item_from_umm_keeps_s30_b8a():
    g = {"id": "HLS.S30.T15TVH.2024186T165849.v2.0", "acquisition": "2024-07-04T17:11:56Z",
         "cloud_cover": 5.0, "platform": "Sentinel-2B",
         "urls": [f"https://data.lpdaac.example/x.{b}.tif" for b in ("B02", "B04", "B8A", "Fmask")]}
    it = hls._item_from_umm(g)
    assert set(it.assets) == {"blue", "red", "nir", "fmask"}
    assert it.assets["nir"].endswith(".B8A.tif") and it.platform == "Sentinel-2B"


def test_scale_and_nodata():
    r = hls.scale_reflectance(np.array([-9999, 0, 1234, 10000], np.int16))
    assert np.isnan(r[0])
    np.testing.assert_allclose(r[1:], [0.0, 0.1234, 1.0], rtol=1e-6)


def test_vegetation_indices_formulas_and_rejection():
    b = np.array([0.04, 0.04, np.nan, 0.04, 0.0, 0.04, 0.2107])
    r = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 1.2, 0.05])
    n = np.array([0.40, 0.40, 0.40, -0.1, 0.40, 0.40, 0.30])
    ndvi, evi, ok = hls.vegetation_indices(b, r, n)
    assert ndvi[0] == pytest.approx((0.40 - 0.05) / 0.45)
    assert evi[0] == pytest.approx(2.5 * 0.35 / (0.40 + 0.30 - 0.30 + 1.0))
    # nodata, negative band, zero band, band > 1, |EVI| > 1 (denominator ~ 0.04)
    assert ok.tolist() == [True, True, False, False, False, False, False]
    assert abs(evi[6]) > 1


def test_block_stats_uses_mask_clear_and_valid():
    mask = np.zeros((4, 4), bool)
    mask[:2] = True  # 8 mask pixels
    fm = np.zeros((4, 4), np.uint8)
    fm[0, 0] = 2  # cloud
    fm[0, 1] = 255  # nodata
    blue = np.full((4, 4), 400, np.int16)
    red = np.full((4, 4), 500, np.int16)
    nir = np.full((4, 4), 4000, np.int16)
    nir[1, 3] = -9999
    s = hls.block_stats(fm, mask, blue, red, nir)
    assert (s["n_mask"], s["n_footprint"], s["n_clear"]) == (8, 7, 5)
    assert s["ndvi_mean"] == pytest.approx(3500 / 4500)
    assert s["ndvi_median"] == pytest.approx(3500 / 4500)
    s0 = hls.block_stats(fm, mask)  # Fmask only (cloudy-skipped granule)
    assert s0["n_clear"] == 6 and np.isnan(s0["ndvi_mean"])


def test_merge_windows_components():
    offs = [(0, 0), (0, 256), (256, 256), (1024, 1024), (2048, 0)]
    g = hls.merge_windows(offs)
    assert len(g) == 3
    w, idx = g[0]
    assert idx == [0, 1, 2] and (w.row_off, w.col_off, w.height, w.width) == (0, 0, 512, 512)
    with pytest.raises(ValueError):
        hls.merge_windows([(10, 0)])


# --- synthetic scene rows ----------------------------------------------------------------
def srow(block, day, ndvi, sensor="L30", n_clear=900, n_mask=1000, state="IA", season=2024,
         tile="T15AAA", gid=None, evi=None):
    d = pd.Timestamp(day)
    return {"granule_id": gid or f"HLS.{sensor}.{tile}.{d:%Y%j}T170000.v2.0", "tile": tile,
            "sensor": sensor, "platform": "x", "acq_dt": d.tz_localize("UTC") + pd.Timedelta(hours=17),
            "cloud_cover": 10.0, "block_id": block, "state": state, "season": season,
            "n_mask": n_mask, "n_footprint": n_mask, "n_clear": n_clear, "ndvi_mean": ndvi,
            "evi_mean": 0.6 * ndvi if evi is None else evi, "ndvi_median": ndvi,
            "refl_read": True}  # fmt: skip


def info(blocks, state="IA", season=2024, n_mask=1000):
    return pd.DataFrame({"season": season, "block_id": blocks, "state": state, "tile": "T15AAA",
                         "n_mask": n_mask})


def test_composite_grid_is_fixed_across_years():
    a = [(d.month, d.day) for d in hls.composite_end_dates(2023)]
    b = [(d.month, d.day) for d in hls.composite_end_dates(2024)]
    assert a == b and a[0] == (5, 16) and a[-1] <= (9, 30) and len(a) == 35
    assert all((y - x).days == 4 for x, y in zip(hls.composite_end_dates(2024),
                                                   hls.composite_end_dates(2024)[1:]))


def test_composite_causality_and_window_edges():
    E = date(2024, 7, 3)
    assert E in hls.composite_end_dates(2024)
    base = [srow("A", E - timedelta(days=15), 0.5), srow("A", E, 0.7)]
    c0 = hls.composite(pd.DataFrame(base), info(["A"]))
    r0 = c0[c0["end_date"] == pd.Timestamp(E)].iloc[0]
    assert r0["ndvi_mean"] == pytest.approx(0.6)  # both edges [E-15, E] included
    assert r0["start_date"] == pd.Timestamp(E - timedelta(days=15))
    assert r0["available_date"] == pd.Timestamp(E + timedelta(days=features.HLS_LAG_DAYS))
    # a scene at E-16 is outside the window ending E
    c1 = hls.composite(pd.DataFrame(base + [srow("A", E - timedelta(days=16), 0.1)]), info(["A"]))
    assert c1.set_index("end_date").loc[pd.Timestamp(E), "ndvi_mean"] == pytest.approx(0.6)
    # scenes at E+1 (and later) never change a composite ending <= E
    c2 = hls.composite(pd.DataFrame(base + [srow("A", E + timedelta(days=1), 0.99),
                                            srow("A", E + timedelta(days=30), 0.2, sensor="S30")]),
                       info(["A"]))
    upto = c0["end_date"] <= pd.Timestamp(E)
    pd.testing.assert_frame_equal(c2[upto.to_numpy()].reset_index(drop=True),
                                  c0[upto].reset_index(drop=True))
    nxt = c2[c2["end_date"] == pd.Timestamp(E + timedelta(days=4))].iloc[0]
    assert nxt["ndvi_mean"] == pytest.approx(np.median([0.7, 0.99]))  # E-15 has left


def test_composite_adding_future_scenes_never_changes_past():
    rng = np.random.default_rng(0)
    days = pd.date_range("2024-05-01", "2024-09-30", freq="D")
    rows = [srow(b, d, float(rng.uniform(0.2, 0.9)), sensor=rng.choice(["L30", "S30"]),
                 n_clear=int(rng.integers(0, 1000)), gid=f"g{b}{i}")
            for i, d in enumerate(days) for b in ("A", "B") if rng.random() < 0.3]
    df = pd.DataFrame(rows)
    full = hls.composite(df, info(["A", "B"]))
    for E in hls.composite_end_dates(2024)[::5]:
        cut = df[pd.to_datetime(df["acq_dt"]).dt.tz_convert(None).dt.normalize() <= pd.Timestamp(E)]
        part = hls.composite(cut, info(["A", "B"]))
        a = full[full["end_date"] <= pd.Timestamp(E)].reset_index(drop=True)
        b = part[part["end_date"] <= pd.Timestamp(E)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        assert (a["obs_date"].dropna() <= a.loc[a["obs_date"].notna(), "end_date"]).all()


def test_block_clear_fraction_gate():
    E = date(2024, 7, 3)
    rows = [srow("A", E, 0.8, n_clear=500), srow("B", E, 0.2, n_clear=499)]
    c = hls.composite(pd.DataFrame(rows), info(["A", "B"]))
    r = c[c["end_date"] == pd.Timestamp(E)].iloc[0]
    assert r["ndvi_mean"] == pytest.approx(0.8) and r["n_blocks"] == 1
    assert r["valid_frac"] == pytest.approx(0.5)


def test_weighted_state_aggregation_missing_blocks_and_valid_frac():
    E = date(2024, 8, 4)
    assert E in hls.composite_end_dates(2024)
    bi = pd.DataFrame({"season": 2024, "block_id": ["A", "B", "C", "D"], "state": "IA",
                       "tile": ["T1", "T1", "T2", "T2"], "n_mask": [1000, 3000, 2000, 4000]})
    rows = [srow("A", E, 0.2, n_clear=1000, n_mask=1000),
            srow("A", E - timedelta(days=3), 0.4, n_clear=1000, n_mask=1000),
            srow("A", E - timedelta(days=6), 0.9, n_clear=1000, n_mask=1000),  # median 0.4
            srow("B", E, 0.6, n_clear=3000, n_mask=3000, sensor="S30"),
            srow("C", E, 0.8, n_clear=2000, n_mask=2000, tile="T2")]  # D missing
    c = hls.composite(pd.DataFrame(rows), bi, cdl_years={("IA", 2024): "2021,2022,2023"})
    r = c[c["end_date"] == pd.Timestamp(E)].iloc[0]
    assert r["ndvi_mean"] == pytest.approx((0.4 * 1000 + 0.6 * 3000 + 0.8 * 2000) / 6000)
    assert r["valid_frac"] == pytest.approx(6000 / 10000)
    assert r["ndvi_median"] == pytest.approx(0.6)  # weights 1000/3000/2000 -> 0.6
    assert (r["n_blocks"], r["n_tiles"], r["n_l30"], r["n_s30"]) == (3, 2, 4, 1)
    assert r["n_pixels"] == 1000 * 3 + 3000 + 2000 and r["cdl_years"] == "2021,2022,2023"
    # empty composites are kept with NaN values and valid_frac 0
    early = c[c["end_date"] == pd.Timestamp(2024, 5, 16)].iloc[0]
    assert np.isnan(early["ndvi_mean"]) and early["valid_frac"] == 0 and pd.isna(early["obs_date"])


def test_obs_date_is_clear_weighted():
    E = date(2024, 7, 3)
    rows = [srow("A", E - timedelta(days=10), 0.5, n_clear=900), srow("B", E, 0.5, n_clear=900),
            srow("B", E - timedelta(days=2), 0.5, n_clear=900, sensor="S30")]
    r = hls.composite(pd.DataFrame(rows), info(["A", "B"])).set_index("end_date").loc[pd.Timestamp(E)]
    assert r["obs_date"] == pd.Timestamp(E - timedelta(days=4))


def test_schema_roundtrip_features(tmp_path, monkeypatch):
    rng = np.random.default_rng(3)
    rows = []
    for y in (2014, 2015, 2016, 2017):
        for i, d in enumerate(pd.date_range(f"{y}-05-01", f"{y}-09-30", freq="2D")):
            for b in ("A", "B"):
                rows.append(srow(b, d, 0.3 + 0.5 * np.sin((d.day_of_year - 120) / 120 * np.pi)
                                 + rng.normal(0, 0.01), season=y, gid=f"g{y}{i}{b}"))
    df = pd.DataFrame(rows)
    bi = pd.concat([info(["A", "B"], season=y) for y in (2014, 2015, 2016, 2017)])
    comps = hls.composite(df, bi, cdl_years={("IA", y): str(y - 1) for y in range(2014, 2018)})
    assert list(comps.columns) == hls.COMPOSITE_COLS
    monkeypatch.setattr(hls, "HLS_STATE_PATH", tmp_path / "hls_state.parquet")
    monkeypatch.setattr(hls, "HLS_PRESAMPLE_PATH", tmp_path / "hls_state_presample.parquet")
    monkeypatch.setattr(features, "HLS_STATE_PATH", tmp_path / "hls_state.parquet")
    monkeypatch.setattr(features, "HLS_PRESAMPLE_PATH", tmp_path / "hls_state_presample.parquet")
    written = hls.write_state(comps)
    assert written == {"hls_state.parquet": 70, "hls_state_presample.parquet": 70}
    h = features.load_hls()
    assert h["presample"].sum() == 70 and len(h) == 140
    p = features.prepare_sat(h, features.HLS_SPEC)
    assert len(p) == int((comps["valid_frac"] >= 0.5).sum())
    assert (p["available_date"] == p["end_date"] + pd.Timedelta(days=features.HLS_LAG_DAYS)).all()
    assert (p["t"] <= p["end_date"]).all()
    blk = features._sat_state(p[p["state"] == "IA"], [2016, 2017], features.HLS_SPEC)
    assert blk["hls_ndvi"].notna().sum() > 100


def test_sensor_bias_synthetic_pairs():
    rows = []
    for y, off in ((2023, 0.02), (2024, -0.01)):
        for k, b in enumerate("ABCDE"):
            d = pd.Timestamp(f"{y}-07-0{k + 1}")
            rows.append(srow(b, d, 0.6 + k / 100, season=y, gid=f"L{y}{b}"))
            rows.append(srow(b, d + pd.Timedelta(days=1), 0.6 + k / 100 + off, sensor="S30",
                             season=y, gid=f"S{y}{b}"))
            # an S30 scene 3 days later must not pair; a cloudy pair must be ignored
            rows.append(srow(b, d + pd.Timedelta(days=3), 0.0, sensor="S30", season=y,
                             gid=f"S3{y}{b}"))
            rows.append(srow(b, d + pd.Timedelta(days=10), 0.1, season=y, gid=f"Lc{y}{b}", n_clear=10))
            rows.append(srow(b, d + pd.Timedelta(days=10), 0.9, sensor="S30", season=y,
                             gid=f"Sc{y}{b}"))
    sb = hls.sensor_bias(pd.DataFrame(rows))
    assert sb.loc[2023, "n_pairs"] == 5 and sb.loc[2024, "n_pairs"] == 5
    assert sb.loc[2023, "median_dndvi"] == pytest.approx(0.02)
    assert sb.loc[2024, "mean_dndvi"] == pytest.approx(-0.01)
    offs = hls.prior_s30_offsets(pd.DataFrame(rows), [2023, 2024, 2025])
    assert offs[2023]["ndvi"] == 0.0  # no prior season: no correction
    assert offs[2024]["ndvi"] == pytest.approx(0.02)  # only 2023 informs 2024
    assert offs[2025]["ndvi"] == pytest.approx(np.median([0.02] * 5 + [-0.01] * 5))


def test_bias_correction_applies_to_s30_only():
    E = date(2024, 7, 3)
    rows = [srow("A", E, 0.6, sensor="S30")]
    c = hls.composite(pd.DataFrame(rows), info(["A"]), s30_offset={2024: {"ndvi": 0.05, "evi": 0.0}})
    assert c.set_index("end_date").loc[pd.Timestamp(E), "ndvi_mean"] == pytest.approx(0.55)
    c = hls.composite(pd.DataFrame([srow("A", E, 0.6)]), info(["A"]),
                      s30_offset={2024: {"ndvi": 0.05, "evi": 0.0}})
    assert c.set_index("end_date").loc[pd.Timestamp(E), "ndvi_mean"] == pytest.approx(0.6)


# --- granule processing and resume (mock reader) -------------------------------------------
def _blocks(n=2):
    m = np.zeros((256, 256), bool)
    m[:128] = True
    return [{"block_id": f"IA_T15AAA_r0000_c{256 * i:04d}", "state": "IA", "season": 2024,
             "row_off": 0, "col_off": 256 * i, "height": 256, "width": 256, "mask": m,
             "n_mask": int(m.sum()), "tile": "T15AAA"} for i in range(n)]


def _item(gid, day, cloudy=False, sensor="L30"):
    return {"id": gid, "tile": "T15AAA", "sensor": sensor, "platform": "LANDSAT-8",
            "acq_dt": pd.Timestamp(day, tz="UTC"), "cloud_cover": 10.0,
            "href_fmask": f"{gid}|fmask|{int(cloudy)}", "href_blue": f"{gid}|blue|0",
            "href_red": f"{gid}|red|0", "href_nir": f"{gid}|nir|0"}


class MockReader:
    def __init__(self):
        self.calls = []

    def __call__(self, href, windows):
        gid, band, cloudy = href.split("|")
        self.calls.append((gid, band, len(windows)))
        val = {"fmask": 2 if cloudy == "1" else 64, "blue": 400, "red": 500, "nir": 4000}[band]
        dt = np.uint8 if band == "fmask" else np.int16
        return [np.full((w.height, w.width), val, dt) for w in windows]


def test_process_granule_reads_bands_only_when_clear():
    rd = MockReader()
    rows, refl = hls.process_granule(_item("g1", "2024-07-01"), _blocks(), rd)
    assert refl and [c[1] for c in rd.calls].count("fmask") == 1
    assert sorted(c[1] for c in rd.calls) == ["blue", "fmask", "nir", "red"]
    assert rd.calls[0][2] == 1  # two adjacent blocks merged into one window
    assert rows[0]["ndvi_mean"] == pytest.approx(3500 / 4500) and rows[0]["n_clear"] == 128 * 256
    rd = MockReader()
    rows, refl = hls.process_granule(_item("g2", "2024-07-02", cloudy=True), _blocks(), rd)
    assert not refl and [c[1] for c in rd.calls] == ["fmask"]
    assert rows[0]["n_clear"] == 0 and not rows[0]["refl_read"]


def test_build_year_resumes_and_skips_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(hls, "SCENES_DIR", tmp_path / "scenes")
    blocks = pd.DataFrame(_blocks())
    monkeypatch.setattr(hls, "season_blocks", lambda year, states=None: blocks)
    items = pd.DataFrame([_item("g1", "2024-06-01"), _item("g2", "2024-06-05", cloudy=True),
                          {**_item("g3", "2024-06-09"), "cloud_cover": 95.0},
                          _item("g4", "2024-10-09")])
    monkeypatch.setattr(hls, "granule_items", lambda year, tile: items)
    rd = MockReader()
    s1 = hls.build_year(2024, ["IA"], workers=2, reader=rd, progress=False)
    assert (s1["granules_found"], s1["granules_prefiltered"]) == (4, 2)
    assert (s1["granules_read"], s1["granules_skipped_cloudy"]) == (1, 1)
    cached = hls.load_scenes(2024, "T15AAA")
    assert len(cached) == 4 and set(cached["granule_id"]) == {"g1", "g2"}
    n_calls = len(rd.calls)
    s2 = hls.build_year(2024, ["IA"], workers=2, reader=rd, progress=False)
    assert len(rd.calls) == n_calls  # nothing re-read, cloudy skip included
    assert s2["granules_cached"] == 2 and s2["granules_todo"] == 0
    # a new block on the same tile triggers reads for that block only
    more = pd.DataFrame(_blocks(3))
    monkeypatch.setattr(hls, "season_blocks", lambda year, states=None: more)
    s3 = hls.build_year(2024, ["IA"], workers=2, reader=rd, progress=False)
    assert s3["granules_todo"] == 2 and len(hls.load_scenes(2024, "T15AAA")) == 6


def test_build_year_failed_granule_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(hls, "SCENES_DIR", tmp_path / "scenes")
    monkeypatch.setattr(hls, "season_blocks", lambda year, states=None: pd.DataFrame(_blocks()))
    monkeypatch.setattr(hls, "granule_items",
                        lambda year, tile: pd.DataFrame([_item("g1", "2024-06-01"),
                                                         _item("bad", "2024-06-02")]))

    def reader(href, windows):
        if href.startswith("bad"):
            raise RuntimeError("COG read failed (RasterioIOError)")
        return MockReader()(href, windows)

    s = hls.build_year(2024, ["IA"], workers=2, reader=reader, progress=False)
    assert s["granules_failed"] == 1 and s["errors"] == {"RuntimeError": 1}
    assert set(hls.load_scenes(2024, "T15AAA")["granule_id"]) == {"g1"}


def test_require_masks_raises_clearly(monkeypatch, tmp_path):
    from cornsat import hlssample

    monkeypatch.setattr(hlssample, "_mask_path", lambda st, season: tmp_path / f"{st}_{season}.npz")
    with pytest.raises(hls.MissingMasksError, match="build_hls_sample.py"):
        hls.require_masks(2031, ["IA"])


def test_write_refuses_symlink(tmp_path):
    real = tmp_path / "real.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(real)
    link = tmp_path / "link.parquet"
    link.symlink_to(real)
    with pytest.raises(RuntimeError, match="symlink"):
        hls._write_atomic(pd.DataFrame({"a": [2]}), link)


def test_weighted_median():
    assert hls.weighted_median([1, 2, 3], [1, 1, 1]) == 2
    assert hls.weighted_median([1, 2, 3], [5, 1, 1]) == 1
    assert np.isnan(hls.weighted_median([np.nan], [1]))


def test_network_process_one_granule():
    try:
        hls.token()
        items = hls.prefilter_items(hls.granule_items(2024, "T15TVH"), 2024)
        it = items[items["cloud_cover"] < 20].iloc[0].to_dict()
        m = np.ones((256, 256), bool)
        b = [{"block_id": "IA_T15TVH_r1792_c2048", "state": "IA", "season": 2024,
              "row_off": 1792, "col_off": 2048, "height": 256, "width": 256, "mask": m}]
        rows, _ = hls.process_granule(it, b)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"network/auth unavailable: {type(e).__name__}")
    assert rows[0]["n_mask"] == 256 * 256 and rows[0]["n_footprint"] > 0

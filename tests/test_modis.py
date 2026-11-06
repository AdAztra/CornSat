from datetime import date

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cornsat import modis
from cornsat.modis import (
    LAG_DAYS,
    apply_qa,
    composite_dates,
    group_composites,
    weighted_median,
    weighted_stats,
)


# --- apply_qa -----------------------------------------------------------------------
def test_apply_qa_fill_cloud_and_scaling():
    ndvi = np.array([[5000, -3000, 8000], [7000, 10000, -2000]], dtype="int16")
    evi = np.array([[3000, 4000, -3000], [5000, 6000, 100]], dtype="int16")
    rel = np.array([[0, 0, 1], [3, 2, 255]], dtype="uint8")
    n, e = apply_qa(ndvi, evi, rel)
    assert n.dtype == np.float32 and e.dtype == np.float32
    assert n[0, 0] == pytest.approx(0.5) and e[0, 0] == pytest.approx(0.3)
    assert np.isnan(n[0, 1])  # NDVI fill
    assert e[0, 1] == pytest.approx(0.4)  # EVI of the same pixel is fine
    assert n[0, 2] == pytest.approx(0.8)  # marginal kept by default
    assert np.isnan(e[0, 2])  # EVI fill
    assert np.isnan(n[1, 0]) and np.isnan(e[1, 0])  # cloudy
    assert np.isnan(n[1, 1]) and np.isnan(e[1, 1])  # snow/ice
    assert np.isnan(n[1, 2]) and np.isnan(e[1, 2])  # reliability fill (uint8 255)


def test_apply_qa_keep_good_only_and_int8_fill():
    ndvi = np.array([5000, 6000, 7000], dtype="int16")
    rel = np.array([0, 1, -1], dtype="int8")
    n, _ = apply_qa(ndvi, ndvi, rel, keep=(0,))
    assert n[0] == pytest.approx(0.5)
    assert np.isnan(n[1]) and np.isnan(n[2])


def test_apply_qa_valid_range():
    raw = np.array([-2000, -2001, 10000, 10001, 0], dtype="int16")
    rel = np.zeros(5, dtype="uint8")
    n, _ = apply_qa(raw, raw, rel)
    assert n[0] == pytest.approx(-0.2) and n[2] == pytest.approx(1.0) and n[4] == 0.0
    assert np.isnan(n[1]) and np.isnan(n[3])


def test_apply_qa_xarray_roundtrip():
    da = xr.DataArray(np.array([[4000, -3000]], dtype="int16"), dims=("y", "x"))
    rel = xr.DataArray(np.array([[0, 0]], dtype="uint8"), dims=("y", "x"))
    n, e = apply_qa(da, da, rel)
    assert isinstance(n, xr.DataArray) and n.dims == ("y", "x")
    assert float(n[0, 0]) == pytest.approx(0.4) and np.isnan(float(n[0, 1]))


# --- weighted stats -----------------------------------------------------------------
def test_weighted_stats_math():
    v = np.array([0.2, 0.8, np.nan, 0.5])
    w = np.array([1.0, 3.0, 2.0, 0.0])
    s = weighted_stats(v, w)
    assert s["mean"] == pytest.approx((0.2 * 1 + 0.8 * 3) / 4)
    assert s["valid_frac"] == pytest.approx(4 / 6)
    assert s["weight"] == pytest.approx(6.0)


def test_weighted_stats_all_nan():
    s = weighted_stats(np.full(4, np.nan), np.ones(4))
    assert np.isnan(s["mean"]) and s["valid_frac"] == 0.0


def test_weighted_stats_zero_weight():
    s = weighted_stats(np.array([0.5, 0.6]), np.zeros(2))
    assert np.isnan(s["mean"]) and np.isnan(s["valid_frac"])


def test_weighted_stats_ignores_bad_weights_and_shape_check():
    s = weighted_stats(np.array([0.5, 0.9, 0.1]), np.array([1.0, np.nan, -2.0]))
    assert s["mean"] == pytest.approx(0.5) and s["valid_frac"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        weighted_stats(np.ones(3), np.ones(4))


def test_weighted_stats_2d_matches_1d():
    rng = np.random.default_rng(0)
    v = rng.random((20, 30))
    v[rng.random((20, 30)) < 0.2] = np.nan
    w = rng.random((20, 30))
    s2, s1 = weighted_stats(v, w), weighted_stats(v.ravel(), w.ravel())
    assert s2["mean"] == pytest.approx(s1["mean"])
    ok = np.isfinite(v)
    assert s2["mean"] == pytest.approx(np.average(v[ok], weights=w[ok]))


def test_weighted_median():
    assert weighted_median([1.0, 2.0, 3.0], [1, 1, 1]) == 2.0
    assert weighted_median([1.0, 2.0, 3.0], [0.1, 0.1, 5.0]) == 3.0
    assert weighted_median([np.nan, 0.4], [10, 1]) == 0.4
    assert np.isnan(weighted_median([np.nan], [1.0]))
    assert np.isnan(weighted_median([0.5], [0.0]))


# --- dates / grouping ---------------------------------------------------------------
def test_composite_dates():
    d = composite_dates(date(2012, 7, 19), date(2012, 8, 3))
    assert d["mid_date"] == date(2012, 7, 26)
    assert d["available_date"] == date(2012, 8, 13)
    assert LAG_DAYS >= 1


def test_available_date_after_end_date_fake_frame():
    rows = []
    for plat, s in [("terra", date(2012, 7, 11)), ("aqua", date(2012, 7, 19)),
                    ("terra", date(2012, 7, 27))]:
        e = s + pd.Timedelta(days=15).to_pytimedelta()
        rows.append({"state": "IA", "year": 2012, "platform": plat, "start_date": s,
                     "end_date": e, **composite_dates(s, e), "obs_date": s, "ndvi_mean": 0.8,
                     "evi_mean": 0.6, "ndvi_mean_good": 0.8, "evi_mean_good": 0.6,
                     "ndvi_median": 0.8, "valid_frac": 1.0, "valid_frac_good": 1.0,
                     "n_tiles": 2, "cdl_years": "2009,2010,2011"})
    df = modis._to_frame(rows)
    assert list(df.columns) == modis.COLUMNS
    assert (df["available_date"] > df["end_date"]).all()
    assert ((df["available_date"] - df["end_date"]).dt.days == LAG_DAYS).all()
    assert (df["mid_date"] > df["start_date"]).all() and (df["mid_date"] < df["end_date"]).all()


class _FakeItem:
    def __init__(self, id_, start, end, platform=None):
        self.id = id_
        self.properties = {"start_datetime": start, "end_datetime": end}
        if platform:
            self.properties["platform"] = platform


def test_group_composites_by_platform_and_window():
    s1, e1 = "2012-07-11T00:00:00Z", "2012-07-26T23:59:59Z"
    s2, e2 = "2012-07-19T00:00:00Z", "2012-08-03T23:59:59Z"
    items = [
        _FakeItem("MOD13Q1.A2012193.h10v04.061.x", s1, e1, "terra"),
        _FakeItem("MOD13Q1.A2012193.h11v04.061.x", s1, e1),  # platform from id
        _FakeItem("MYD13Q1.A2012201.h10v04.061.x", s2, e2, "aqua"),
    ]
    g = group_composites(items)
    assert [(p, s, len(its)) for p, s, _, its in g] == [
        ("terra", date(2012, 7, 11), 2),
        ("aqua", date(2012, 7, 19), 1),
    ]
    assert g[0][2] == date(2012, 7, 26)


# --- network ------------------------------------------------------------------------
def test_load_one_ia_composite_network():  # skips itself when offline
    try:
        items = [i for i in modis.search_items("IA", 2012)
                 if i.properties["start_datetime"].startswith("2012-07-19")]
        if not items:
            pytest.skip("no items returned")
        ds = modis.load_composite(items, "IA").compute(scheduler="threads", num_workers=4)
    except Exception as e:  # offline
        pytest.skip(f"Planetary Computer unavailable: {e}")
    from cornsat.cornmask import modis_geobox

    assert ds["ndvi"].shape == tuple(modis_geobox("IA").shape)
    n, _ = apply_qa(ds["ndvi"].values, ds["evi"].values, ds["reliability"].values)
    assert 0.5 < np.nanmean(n) < 0.95

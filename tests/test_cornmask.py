import numpy as np
import pytest

from cornsat import cornmask
from cornsat.cornmask import (
    MODIS_ORIGIN_X,
    MODIS_ORIGIN_Y,
    MODIS_RES,
    mask_years,
    modis_geobox,
)


@pytest.mark.parametrize("season", range(2009, 2027))
def test_mask_years_no_lookahead(season):
    ys = mask_years(season)
    assert 1 <= len(ys) <= 3
    assert all(y < season for y in ys)
    assert all(season - 3 <= y <= season - 1 for y in ys)
    assert all(y >= 2008 for y in ys)
    assert ys == sorted(ys)


def test_mask_years_2009_and_full_window():
    assert mask_years(2009) == [2008]
    assert mask_years(2010) == [2008, 2009]
    assert mask_years(2012) == [2009, 2010, 2011]
    assert mask_years(2026) == [2023, 2024, 2025]


@pytest.mark.parametrize("state", ["IA", "IL", "NE", "MN", "IN"])
def test_modis_geobox_sinusoidal_and_aligned(state):
    gb = modis_geobox(state)
    assert "sinusoidal" in gb.crs.to_wkt().lower()
    aff = gb.affine
    assert aff.a == pytest.approx(MODIS_RES) and aff.e == pytest.approx(-MODIS_RES)
    ix = (aff.c - MODIS_ORIGIN_X) / MODIS_RES
    iy = (MODIS_ORIGIN_Y - aff.f) / MODIS_RES
    assert ix == pytest.approx(round(ix), abs=1e-6)
    assert iy == pytest.approx(round(iy), abs=1e-6)


def test_modis_geobox_covers_state():
    from cornsat.aoi import load_states

    gb = modis_geobox("IA")
    minx, miny, maxx, maxy = load_states("19").to_crs(gb.crs.to_wkt()).total_bounds
    l, b, r, t = gb.boundingbox
    assert l <= minx and b <= miny and r >= maxx and t >= maxy


def test_corn_fraction_range_and_area():
    try:
        da = cornmask.corn_fraction("IA", 2012)
    except Exception as e:  # network unavailable and nothing cached
        pytest.skip(f"CDL unavailable: {e}")
    v = da.values
    assert v.shape == tuple(modis_geobox("IA").shape)
    assert np.isfinite(v).all()
    assert v.min() >= 0.0 and v.max() <= 1.0
    assert v.max() > 0.5
    acres = cornmask.corn_acres(da) / 1e6
    assert 12.5 < acres < 14.5  # Iowa corn, million acres


def test_season_weight_attrs():
    try:
        w = cornmask.season_weight("IA", 2013)
    except Exception as e:
        pytest.skip(f"CDL unavailable: {e}")
    assert w.attrs["cdl_years"] == [2010, 2011, 2012]
    assert w.attrs["n_years"] == 3
    assert 0.0 <= float(w.min()) and float(w.max()) <= 1.0

"""Phase 5 feature tests. The most important is the truncation test: rebuilding from
inputs truncated at date d (by availability date) must reproduce the full-build values
at d exactly, which proves that no future information reaches a feature."""

import json
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cornsat import features as F
from cornsat import usda, weather
from cornsat.config import DATA_PROCESSED, STATES

FX = Path(__file__).parent / "fixtures"
REAL = all((DATA_PROCESSED / f).exists() for f in
           ("modis_state.parquet", "weather_daily.parquet", "zcz_front_dec.parquet"))
needs_real = pytest.mark.skipif(not REAL, reason="processed data not built")
SYN_YEARS = range(2003, 2015)
SYN_STATES = list(STATES)


# --- synthetic inputs ---------------------------------------------------------------
def synth_modis(years=SYN_YEARS, states=SYN_STATES, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for st_i, st in enumerate(states):
        for y in years:
            lvl = rng.normal(0, 0.03)
            for plat, off in (("terra", 0), ("aqua", 8)):
                for k in range(23):
                    start = date(y, 1, 1) + timedelta(days=16 * k + off)
                    end = start + timedelta(days=15)
                    if end < date(y, 4, 1) or start > date(y, 10, 31):
                        continue
                    obs = start + timedelta(days=int(rng.integers(3, 13)))
                    doy = obs.timetuple().tm_yday
                    v = 0.25 + 0.55 * np.exp(-(((doy - 205) / 45) ** 2)) + lvl + st_i * 0.01
                    v += rng.normal(0, 0.01)
                    rows.append({
                        "state": st, "year": y, "platform": plat,
                        "start_date": pd.Timestamp(start), "end_date": pd.Timestamp(end),
                        "mid_date": pd.Timestamp(start + timedelta(days=7)),
                        "available_date": pd.Timestamp(end + timedelta(days=10)),
                        "obs_date": pd.Timestamp(obs), "ndvi_mean": v, "evi_mean": 0.8 * v,
                        "valid_frac": 0.3 if rng.random() < 0.03 else 0.97,
                    })  # fmt: skip
    return pd.DataFrame(rows)


def synth_weather(years=SYN_YEARS, states=SYN_STATES, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D")
    doy = dates.day_of_year.to_numpy()
    parts = []
    for st in states:
        tmax = 15 + 16 * np.sin((doy - 105) / 365 * 2 * np.pi) + rng.normal(0, 4, len(dates))
        d = pd.DataFrame({"date": dates, "tmax_c": tmax, "tmin_c": tmax - 11,
                          "precip_mm": rng.gamma(0.5, 5, len(dates))})
        d = weather.add_features(d)
        d.insert(0, "state", st)
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def _rec(stat, unit, geo, y, we, v):
    return {"source_desc": "SURVEY", "commodity_desc": "CORN", "statisticcat_desc": stat,
            "unit_desc": unit, "agg_level_desc": "NATIONAL" if geo == "US" else "STATE",
            "state_alpha": geo, "year": str(y), "week_ending": we.isoformat(),
            "freq_desc": "WEEKLY", "Value": f"{v:.6f}"}


def ge_target(geo, y, doy):
    return 50 + (y - 2000) * 0.7 + (sum(map(ord, geo)) % 5) + 0.05 * doy + (3 if y % 3 == 0 else 0)


def silk_target(y, doy):
    return 2.0 * (doy - 170) + (y % 4) * 2.0


def synth_usda(years, geos=("US", *SYN_STATES)) -> dict:
    """Raw NASS-style records parsed through usda.parse_condition / parse_progress."""
    cond, prog = [], []
    for y in years:
        d = date(y, 4, 1)
        d += timedelta(days=(6 - d.weekday()) % 7)  # first Sunday
        while d <= date(y, 11, 30):
            doy = int(F.cal_doy([d])[0])
            for g in geos:
                if date(y, 6, 1) <= d <= date(y, 10, 5):
                    ge = ge_target(g, y, doy)
                    g_, e_ = 0.7 * ge, 0.3 * ge
                    rest = 100 - ge
                    for unit, v in (("PCT VERY POOR", 0.15 * rest), ("PCT POOR", 0.25 * rest),
                                    ("PCT FAIR", 0.6 * rest), ("PCT GOOD", g_),
                                    ("PCT EXCELLENT", e_)):
                        cond.append(_rec("CONDITION", unit, g, y, d, v))
                prog.append(_rec("PROGRESS", "PCT PLANTED", g, y, d, min(100, 3 * (doy - 100))))
                if 175 <= doy <= 220:
                    prog.append(_rec("PROGRESS", "PCT SILKING", g, y, d, silk_target(y, doy)))
            d += timedelta(days=7)
    return {"condition": usda.parse_condition(pd.DataFrame(cond)),
            "progress": usda.parse_progress(pd.DataFrame(prog))}


def truncate(inputs: dict, d: pd.Timestamp) -> dict:
    """Keep only input rows whose availability date is <= d (every table, including the
    prior-season climatology history)."""
    m = inputs["modis"]
    out = {"modis": m[F.modis_available_date(m) <= d],
           "weather": inputs["weather"][pd.to_datetime(inputs["weather"]["available_date"]) <= d],
           "usda": None}
    if inputs.get("usda"):
        out["usda"] = {k: v[pd.to_datetime(v["available_date"]) <= d]
                       for k, v in inputs["usda"].items()}
    return out


def assert_rows_equal(a: pd.Series, b: pd.Series):
    assert list(a.index) == list(b.index)
    for c in a.index:
        x, y = a[c], b[c]
        if pd.isna(x) and pd.isna(y):
            continue
        assert x == y, f"{c}: full={x!r} truncated={y!r}"


# --- fixtures -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def syn_inputs():
    return {"modis": synth_modis(), "weather": synth_weather(),
            "usda": synth_usda(range(2006, 2015))}


@pytest.fixture(scope="module")
def syn_panel(syn_inputs):
    i = syn_inputs
    return F.build_features(i["modis"], i["weather"], i["usda"], seasons=range(2009, 2015))


@pytest.fixture(scope="module")
def real_inputs():
    inp = F.load_inputs(presample=True)
    inp["usda"] = inp["usda"] or synth_usda(range(2009, 2027))  # exercise the USDA path
    return inp


@pytest.fixture(scope="module")
def real_panel(real_inputs):
    i = real_inputs
    return F.build_features(i["modis"], i["weather"], i["usda"])


# --- helpers ------------------------------------------------------------------------
def test_cal_doy_folds_leap_day():
    d = pd.to_datetime(["2012-03-01", "2011-03-01", "2012-12-31", "2011-12-31", "2012-02-28"])
    assert list(F.cal_doy(d)) == [60, 60, 365, 365, 59]


# --- (a) as-of columns <= date ------------------------------------------------------
def _check_asof(panel):
    for c in ("modis_asof", "weather_asof", "usda_asof"):
        assert c in panel
        s = panel[c]
        assert (s.isna() | (s <= panel["date"])).all(), c
        assert s.notna().any(), c


def test_asof_le_date_synthetic(syn_panel):
    _check_asof(syn_panel)


@needs_real
def test_asof_le_date_real(real_panel):
    _check_asof(real_panel)
    assert set(real_panel["geo"]) == set(F.GEOS)
    assert real_panel.groupby("geo").size().nunique() == 1
    days = real_panel["date"]
    assert ((days.dt.month >= 4) & (days.dt.month <= 10)).all()


# --- (b) truncation test ------------------------------------------------------------
def _truncation(inputs, full, pairs):
    for geo, d in pairs:
        d = pd.Timestamp(d)
        tr = truncate(inputs, d)
        assert len(tr["modis"]) < len(inputs["modis"])  # something future was removed
        assert len(tr["weather"]) < len(inputs["weather"])
        part = F.build_features(tr["modis"], tr["weather"], tr["usda"], seasons=[d.year])
        a = full[(full["geo"] == geo) & (full["date"] == d)].iloc[0]
        b = part[(part["geo"] == geo) & (part["date"] == d)].iloc[0]
        assert_rows_equal(a, b)
        # the whole season-to-date of every geo must match too, not only the sampled row
        cols = [c for c in full.columns]
        fa = full[(full["date"].dt.year == d.year) & (full["date"] <= d)][cols]
        fb = part[part["date"] <= d][cols]
        pd.testing.assert_frame_equal(fa.reset_index(drop=True), fb.reset_index(drop=True),
                                      check_exact=True)


def test_truncation_synthetic(syn_inputs, syn_panel):
    rng = np.random.default_rng(7)
    pairs = []
    for y in (2009, 2011, 2014):
        days = pd.date_range(f"{y}-04-01", f"{y}-10-31")
        pairs.append((F.GEOS[rng.integers(len(F.GEOS))], days[rng.integers(len(days))]))
    _truncation(syn_inputs, syn_panel, pairs)


@needs_real
def test_truncation_real(real_inputs, real_panel):
    rng = np.random.default_rng(2024)
    pairs = [("US", "2012-07-25"), ("IA", "2025-08-05"), ("IL", "2026-10-20"),
             ("NE", "2009-06-15")]
    for y in rng.choice(np.arange(2010, 2025), size=4, replace=False):
        days = pd.date_range(f"{y}-04-01", f"{y}-10-31")
        pairs.append((F.GEOS[rng.integers(len(F.GEOS))], days[rng.integers(len(days))]))
    _truncation(real_inputs, real_panel, pairs)


# --- (c) climatology never uses season Y or later -----------------------------------
def test_climatology_prior_seasons_only():
    grid = F.DOY_GRID
    curves = pd.DataFrame({y: np.full(len(grid), float(y)) for y in range(2003, 2020)}).T
    curves.columns = grid
    for y in (2006, 2009, 2015):
        c = F.climatology(curves, y)
        assert c["mean"].iloc[100] == pytest.approx(np.mean(range(2003, y)))
        assert c["n"].iloc[100] == y - 2003
        assert max(F.climatology_years(curves, y)) == y - 1
    # fewer than MIN_CLIM_YEARS prior seasons -> NaN
    assert F.climatology(curves, 2005)["mean"].isna().all()


def test_climatology_ignores_current_and_future_seasons(syn_inputs):
    m = F.prepare_modis(syn_inputs["modis"])
    m = m[m["state"] == "IA"]
    base = F.climatology(F.season_curves(m, "ndvi_mean"), 2010)
    bumped = m.copy()
    bumped.loc[bumped["season"] >= 2010, "ndvi_mean"] += 5.0
    after = F.climatology(F.season_curves(bumped, "ndvi_mean"), 2010)
    pd.testing.assert_frame_equal(base, after)
    nxt = F.climatology(F.season_curves(bumped, "ndvi_mean"), 2011)
    assert nxt["mean"].loc[200] - base["mean"].loc[200] > 0.5  # 2010 enters for 2011


@pytest.mark.skipif(not (DATA_PROCESSED / "modis_state_presample.parquet").exists(),
                    reason="pre-sample MODIS not built")
@needs_real
def test_real_climatology_baseline_years(real_inputs):
    m = F.prepare_modis(real_inputs["modis"])
    curves = F.season_curves(m[m["state"] == "IA"], "ndvi_mean")
    assert F.climatology_years(curves, 2009) == list(range(2003, 2009))
    assert F.climatology_years(curves, 2013) == list(range(2003, 2013))


# --- (d) US weights -----------------------------------------------------------------
def test_us_weights_sum_to_one():
    for k in range(1, len(STATES) + 1):
        for sub in combinations(STATES, k):
            w = F.us_weights(sub)
            assert set(w) == set(sub)
            assert sum(w.values()) == pytest.approx(1.0)
    assert sum(F.US_WEIGHTS.values()) == pytest.approx(1.0)


def test_us_aggregate_renormalises(syn_panel):
    p = syn_panel
    d = p[(p["geo"] != "US") & p["ndvi_anom"].notna()]["date"].iloc[500]
    rows = p[p["date"] == d].set_index("geo")
    have = [s for s in STATES if pd.notna(rows.loc[s, "ndvi_anom"])]
    w = F.us_weights(have)
    exp = sum(w[s] * rows.loc[s, "ndvi_anom"] for s in have)
    assert rows.loc["US", "ndvi_anom"] == pytest.approx(exp)


# --- (e) MODIS staleness ------------------------------------------------------------
def _check_stale(panel):
    stale = panel["modis_age_days"] > F.MAX_STALENESS_DAYS
    assert panel.loc[stale, F.MODIS_VALUE_COLS].isna().all().all()


def test_staleness_synthetic(syn_inputs):
    m = syn_inputs["modis"]
    m = m[~((m["year"] == 2012) & (m["start_date"] >= "2012-07-01"))]  # outage
    p = F.build_features(m, syn_inputs["weather"], None, seasons=[2012])
    _check_stale(p)
    aug = p[(p["date"] >= "2012-08-20") & (p["date"] <= "2012-08-31")]
    assert aug[F.MODIS_VALUE_COLS].isna().all().all()
    jul = p[(p["date"] == "2012-07-10") & (p["geo"] == "IA")]
    assert jul["ndvi_anom"].notna().all()


@needs_real
def test_staleness_2025_august(real_panel):
    p = real_panel
    _check_stale(p)
    s25 = p[p["season"] == 2025]
    last_end = pd.Timestamp("2025-07-11")  # last 2025 composite before the PC outage
    cutoff = last_end + pd.Timedelta(days=F.MAX_STALENESS_DAYS)
    late = s25[(s25["date"] > cutoff) & (s25["date"] <= "2025-10-31")]
    assert late[F.MODIS_VALUE_COLS].isna().all().all()
    assert (late["modis_age_days"] > F.MAX_STALENESS_DAYS).all()
    aug = s25[s25["date"].dt.month == 8]
    assert aug.loc[aug["date"] > cutoff, "ndvi_anom"].isna().all()
    # still fresh at the start of August (latest composite ended Jul 11)
    assert s25.loc[s25["date"] == "2025-08-01", "ndvi_anom"].notna().all()
    # weather is unaffected
    assert late["gdd_cum_anom"].notna().all()


# --- (f) USDA path ------------------------------------------------------------------
def _load_fx(name):
    with open(FX / name) as f:
        return pd.DataFrame(json.load(f)["data"])


def test_usda_fixture_path(syn_inputs):
    u = {"condition": usda.parse_condition(_load_fx("usda_condition.json")),
         "progress": usda.parse_progress(_load_fx("usda_progress.json"))}
    p = F.build_features(syn_inputs["modis"], syn_inputs["weather"], u, seasons=[2012])
    assert "usda_asof" in p and "ge" in p and "silking_anom" in p
    _check_asof(p)
    for geo in ("US", "IA"):
        exp = u["condition"].query("geo == @geo").iloc[0]
        g = p[p["geo"] == geo].set_index("date")
        avail = exp["available_date"]  # Tue 2012-07-31
        assert avail == pd.Timestamp("2012-07-31")
        assert np.isnan(g.loc[avail - pd.Timedelta(days=1), "ge"])
        assert g.loc[avail, "ge"] == pytest.approx(exp["ge"])
        assert g.loc[avail, "ci"] == pytest.approx(exp["ci"])
        assert g.loc[avail, "usda_asof"] == avail
        stale = exp["week_ending"] + pd.Timedelta(days=F.USDA_MAX_STALENESS_DAYS + 1)
        assert np.isnan(g.loc[stale, "ge"])
        assert np.isnan(g.loc[avail, "ge_anom"])  # only one season: no climatology
    assert p.loc[p["geo"] == "MN", "ge"].isna().all()


def test_usda_absent_gives_no_usda_columns(syn_inputs):
    p = F.build_features(syn_inputs["modis"], syn_inputs["weather"], None, seasons=[2012])
    assert "usda_asof" not in p and "ge" not in p
    assert p["ndvi_anom"].notna().any() and p["gdd_cum_anom"].notna().any()


def test_usda_anomalies_synthetic(syn_panel):
    p = syn_panel
    y, geo = 2013, "IA"
    g = p[(p["geo"] == geo) & (p["season"] == y)].set_index("date")
    d = pd.Timestamp("2013-07-17")  # Wed: newest week ending is Sun 2013-07-14
    we = pd.Timestamp("2013-07-14")
    doy = int(F.cal_doy([we])[0])
    assert g.loc[d, "usda_asof"] == pd.Timestamp("2013-07-16")
    assert g.loc[d, "ge"] == pytest.approx(ge_target(geo, y, doy))
    # synthetic ge is linear in doy within a season, so the interpolated climatology at doy
    # is the mean over prior seasons (2006-2012) of ge_target(., s, doy)
    clim = np.mean([ge_target(geo, s, doy) for s in range(2006, y)])
    assert g.loc[d, "ge_anom"] == pytest.approx(ge_target(geo, y, doy) - clim)
    assert g.loc[d, "ge_chg_2w"] == pytest.approx(0.05 * 14)
    sclim = np.mean([silk_target(s, doy) for s in range(2006, y)])
    assert g.loc[d, "pct_silking"] == pytest.approx(silk_target(y, doy))
    assert g.loc[d, "silking_anom"] == pytest.approx(silk_target(y, doy) - sclim)
    # before the first silking report (and before Jul 1) silking counts as 0
    assert g.loc[pd.Timestamp("2013-06-12"), "pct_silking"] == 0.0


# --- weather definitions ------------------------------------------------------------
def test_weather_definitions(syn_inputs, syn_panel):
    w = syn_inputs["weather"]
    ia = w[w["state"] == "IA"].set_index("date")
    g = syn_panel[syn_panel["geo"] == "IA"].set_index("date")
    d = pd.Timestamp("2011-07-20")
    last = d - pd.Timedelta(days=1)
    assert g.loc[d, "weather_asof"] == d
    assert g.loc[d, "edd30_jul"] == pytest.approx(ia.loc["2011-07-01":last, "edd30"].sum())
    assert g.loc[d, "edd30_14d"] == pytest.approx(
        ia.loc[last - pd.Timedelta(days=13):last, "edd30"].sum())
    assert g.loc[d, "edd30_cum"] == pytest.approx(ia.loc[last, "edd30_cum"])
    doy = int(F.cal_doy([last])[0])
    prior = [ia.loc[ia.index[(ia.index.year == s) & (F.cal_doy(ia.index) == doy)], "gdd_cum"].mean()
             for s in range(2003, 2011)]
    assert g.loc[d, "gdd_cum_anom"] == pytest.approx(ia.loc[last, "gdd_cum"] - np.mean(prior))
    aug = pd.Timestamp("2011-08-20")
    assert g.loc[aug, "edd30_jul"] == pytest.approx(ia.loc["2011-07-01":"2011-07-31", "edd30"].sum())


# --- trading table ------------------------------------------------------------------
@needs_real
def test_trading_table(real_panel):
    zcz = pd.read_parquet(DATA_PROCESSED / "zcz_front_dec.parquet")
    t = F.build_trading(real_panel, zcz)
    assert (t["campaign"] == t["date"].dt.year).all()
    assert t["date"].is_unique
    assert "ndvi_anom" in t and "ndvi_anom_ia" in t and "edd30_jul_ne" in t
    assert (t["modis_asof"].isna() | (t["modis_asof"] <= t["date"])).all()

"""Phase 5 feature tests. The most important is the truncation test: rebuilding from
inputs truncated at date d (by availability date) must reproduce the full-build values
at d exactly, which proves that no future information reaches a feature."""

import hashlib
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


HLS_SYN_PRESAMPLE = range(2013, 2016)
HLS_SYN_YEARS = range(2016, 2021)


def synth_hls(years=HLS_SYN_YEARS, states=SYN_STATES, seed=5, presample=False) -> pd.DataFrame:
    """HLS-style composites: a 16-day trailing window ending every 4 days (consecutive
    composites overlap by 12 days), platform "hls", no ``*_good`` columns. obs_date is
    fixed at end - 7 so composites are strictly ordered in time (4 days apart). The stored
    available_date (end + 3) is deliberately wrong: the panel must use end + HLS_LAG_DAYS."""
    rng = np.random.default_rng(seed)
    rows = []
    for st_i, st in enumerate(states):
        for y in years:
            lvl = rng.normal(0, 0.03)
            end = date(y, 3, 1)
            while end <= date(y, 11, 30):
                start = end - timedelta(days=15)
                obs = end - timedelta(days=7)
                doy = obs.timetuple().tm_yday
                v = 0.22 + 0.6 * np.exp(-(((doy - 200) / 42) ** 2)) + lvl + st_i * 0.01
                v += rng.normal(0, 0.01)
                rows.append({
                    "state": st, "year": y, "platform": "hls",
                    "start_date": pd.Timestamp(start), "end_date": pd.Timestamp(end),
                    "mid_date": pd.Timestamp(start + timedelta(days=7)),
                    "obs_date": pd.Timestamp(obs),
                    "available_date": pd.Timestamp(end + timedelta(days=3)),
                    "ndvi_mean": v, "evi_mean": 0.75 * v, "ndvi_median": v - 0.01,
                    "valid_frac": 0.2 if rng.random() < 0.04 else 0.9,
                    "n_tiles": 4, "n_blocks": 40, "n_pixels": 10_000, "n_l30": 3, "n_s30": 5,
                    "cdl_years": str(y),
                })  # fmt: skip
                end += timedelta(days=4)
    return pd.DataFrame(rows).assign(presample=presample)


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


def synth_zcz(years) -> pd.DataFrame:
    """Business-day front-December table (Dec 15 of y-1 .. Dec 14 of y -> campaign y)."""
    rows = []
    for y in years:
        expiry = pd.Timestamp(f"{y}-12-14")
        for d in pd.bdate_range(f"{y - 1}-12-15", expiry):
            rows.append({"date": d, "contract": f"ZCZ{y % 100:02d}", "campaign": y,
                         "close": 400.0 + (d - pd.Timestamp(f"{y}-01-01")).days * 0.1,
                         "ret": 0.0, "days_to_expiry": float((expiry - d).days)})
    return pd.DataFrame(rows)


def frame_digest(df: pd.DataFrame) -> str:
    """sha256 over column names + dtypes (order included) and the per-row hash of every
    value and of the index. NaNs are canonicalised first, so the digest pins values and
    layout, not NaN bit patterns. Float results are bit-exact, so the pinned digests hold
    for the platform / locked numpy+pandas they were taken on (macOS arm64, uv.lock)."""
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].where(d[c].notna(), np.nan)
    h = hashlib.sha256()
    h.update(repr([(str(c), str(t)) for c, t in d.dtypes.items()]).encode())
    h.update(pd.util.hash_pandas_object(d, index=True).to_numpy().tobytes())
    return h.hexdigest()


def truncate(inputs: dict, d: pd.Timestamp) -> dict:
    """Keep only input rows whose availability date is <= d (every table, including the
    prior-season climatology history)."""
    m = inputs["modis"]
    out = {"modis": m[F.modis_available_date(m) <= d],
           "weather": inputs["weather"][pd.to_datetime(inputs["weather"]["available_date"]) <= d],
           "usda": None, "hls": None}
    if inputs.get("usda"):
        out["usda"] = {k: v[pd.to_datetime(v["available_date"]) <= d]
                       for k, v in inputs["usda"].items()}
    h = inputs.get("hls")
    if h is not None:  # pre-sample rows too: availability, not the flag, decides
        out["hls"] = h[F.sat_available_date(h, F.HLS_SPEC) <= d]
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
def syn_hls_inputs():
    """Seasons 2010-2020 of MODIS/weather/USDA plus HLS: 2013-2015 pre-sample (flagged,
    climatology only) and 2016-2020 sample."""
    yrs = range(2010, 2021)
    hls = pd.concat([synth_hls(HLS_SYN_PRESAMPLE, seed=4, presample=True),
                     synth_hls(HLS_SYN_YEARS)], ignore_index=True)
    return {"modis": synth_modis(yrs), "weather": synth_weather(yrs),
            "usda": synth_usda(range(2013, 2021)), "hls": hls}


HLS_SYN_SEASONS = range(2015, 2021)  # 2015: a pre-sample season -> HLS columns all NaN


@pytest.fixture(scope="module")
def syn_hls_panel(syn_hls_inputs):
    i = syn_hls_inputs
    return F.build_features(i["modis"], i["weather"], i["usda"], seasons=HLS_SYN_SEASONS,
                            hls=i["hls"])


@pytest.fixture(scope="module")
def real_inputs():
    inp = F.load_inputs(presample=True)
    inp["usda"] = inp["usda"] or synth_usda(range(2009, 2027))  # exercise the USDA path
    return inp


@pytest.fixture(scope="module")
def real_panel(real_inputs):
    i = real_inputs
    return F.build_features(i["modis"], i["weather"], i["usda"], hls=i.get("hls"))


# --- helpers ------------------------------------------------------------------------
def test_cal_doy_folds_leap_day():
    d = pd.to_datetime(["2012-03-01", "2011-03-01", "2012-12-31", "2011-12-31", "2012-02-28"])
    assert list(F.cal_doy(d)) == [60, 60, 365, 365, 59]


# --- characterisation: MODIS-only outputs are bit-identical to the pre-HLS code -----------
# Digests taken with the Phase 7 code (before SatSpec / HLS) on the same inputs. They pin
# every value, dtype, column name and column order of the panel and the trading table.
SYN_PANEL_DIGEST = "a4bb4a92e52235316181c68b22ee69d1d6fc0d0d1b58e58b83dbac923a60c3e4"
SYN_TRADING_DIGEST = "12c8a2467c585db4a5e93b019541c85c98458a90f930ab361dd591b8362e3027"
REAL_INPUT_DIGEST = "c68736494b1eb26a786ff18c917e201267af187916470b1e0ca192f5e4df9a06"
REAL_PANEL_DIGEST = "342cf8c9c5f4f04d84e73bbf44e039d50aea4c1ac68f1ec6561920b107bad573"
REAL_TRADING_DIGEST = "96604334ecdf9a30f028f4a32533fc19fe940004cc74ec41148a7743cb2763a2"


REAL_INPUT_FILES = ("modis_state", "modis_state_presample", "weather_daily",
                    "weather_daily_presample", "usda_condition", "usda_progress",
                    "zcz_front_dec")


def real_input_digest() -> str:
    """sha256 of the bytes of the processed files the MODIS-path real build reads. The
    output pins are only meaningful for exactly these inputs (a rebuild of the data, not
    of the code, changes them)."""
    h = hashlib.sha256()
    for name in REAL_INPUT_FILES:
        p = DATA_PROCESSED / f"{name}.parquet"
        h.update(name.encode())
        h.update(p.read_bytes() if p.exists() else b"<missing>")
    return h.hexdigest()


def test_modis_output_unchanged_synthetic(syn_panel):
    assert frame_digest(syn_panel) == SYN_PANEL_DIGEST
    trading = F.build_trading(syn_panel, synth_zcz(range(2009, 2015)))
    assert frame_digest(trading) == SYN_TRADING_DIGEST


@needs_real
def test_modis_output_unchanged_real(real_inputs, real_panel):
    if real_input_digest() != REAL_INPUT_DIGEST:
        pytest.skip("real processed inputs changed since the digests were pinned")
    zcz = pd.read_parquet(DATA_PROCESSED / "zcz_front_dec.parquet")
    # real_panel carries the HLS block once hls_state.parquet exists: without it, the
    # panel must still be exactly the MODIS-only one
    p = real_panel[[c for c in real_panel.columns if c not in F.HLS_SPEC.block_cols]]
    assert frame_digest(p) == REAL_PANEL_DIGEST
    assert frame_digest(F.build_trading(p, zcz)) == REAL_TRADING_DIGEST


# --- (a) as-of columns <= date ------------------------------------------------------
def _check_asof(panel):
    cols = ["modis_asof", "weather_asof", "usda_asof"]
    cols += [F.HLS_SPEC.asof_col] if F.HLS_SPEC.asof_col in panel else []
    for c in cols:
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
        if inputs.get("hls") is not None:
            assert len(tr["hls"]) < len(inputs["hls"])
            assert F.HLS_SPEC.asof_col in full
        part = F.build_features(tr["modis"], tr["weather"], tr["usda"], seasons=[d.year],
                                hls=tr["hls"])
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


def test_truncation_synthetic_hls(syn_hls_inputs, syn_hls_panel):
    """Same no-look-ahead proof with overlapping 4-day HLS composites (HLS columns
    included in the row and season-to-date comparisons, exact)."""
    rng = np.random.default_rng(11)
    pairs = [("IA", "2015-08-01"), ("US", "2016-04-02"), ("NE", "2016-07-19"),
             ("US", "2020-10-31")]  # pre-sample season, first sample days, mid, last day
    for y in (2017, 2018, 2019):
        days = pd.date_range(f"{y}-04-01", f"{y}-10-31")
        pairs.append((F.GEOS[rng.integers(len(F.GEOS))], days[rng.integers(len(days))]))
    _truncation(syn_hls_inputs, syn_hls_panel, pairs)


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
    last = d - pd.Timedelta(days=weather.WEATHER_LAG_DAYS)  # newest weather day at d is d-3
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


def test_weather_three_day_lag_in_panel(syn_inputs, syn_panel):
    """Row d sees weather only through d-3 (NASA POWER lag), never d-1 or d-2."""
    assert weather.WEATHER_LAG_DAYS == 3
    assert F.WEATHER_MAX_STALENESS_DAYS >= weather.WEATHER_LAG_DAYS
    w = syn_inputs["weather"]
    ia = w[w["state"] == "IA"].set_index("date")
    g = syn_panel[syn_panel["geo"] == "IA"].set_index("date")
    d = pd.Timestamp("2011-07-20")
    d3, d1 = d - pd.Timedelta(days=3), d - pd.Timedelta(days=1)
    assert g.loc[d, "edd30_cum"] == pytest.approx(ia.loc[d3, "edd30_cum"])
    assert g.loc[d, "edd30_cum"] != pytest.approx(ia.loc[d1, "edd30_cum"])
    assert (g["weather_asof"].dropna() <= g["weather_asof"].dropna().index.to_series()).all()
    # a weather outage: rows are blank once the newest day is older than the staleness limit
    ww = w[~((w["date"] >= "2011-07-01") & (w["date"] <= "2011-07-31"))]
    p = F.build_features(syn_inputs["modis"], ww, None, seasons=[2011])
    gi = p[p["geo"] == "IA"].set_index("date")
    stale_from = pd.Timestamp("2011-06-30") + pd.Timedelta(days=F.WEATHER_MAX_STALENESS_DAYS + 1)
    assert pd.notna(gi.loc[stale_from - pd.Timedelta(days=1), "gdd_cum"])
    assert np.isnan(gi.loc[stale_from, "gdd_cum"])


# --- US aggregate composition -------------------------------------------------------
def test_us_n_states_columns(syn_inputs):
    m = syn_inputs["modis"]
    m = m[m["state"] != "IL"]  # one state never has MODIS
    w = syn_inputs["weather"]
    w = w[~((w["state"] == "NE") & (w["date"] >= "2012-07-01") & (w["date"] <= "2012-07-31"))]
    p = F.build_features(m, w, None, seasons=[2012])
    us = p[p["geo"] == "US"].set_index("date")
    st = p[p["geo"] != "US"]
    exp_m = st.groupby("date")["ndvi_anom"].apply(lambda s: s.notna().sum())
    exp_w = st.groupby("date")["gdd_cum_anom"].apply(lambda s: s.notna().sum())
    pd.testing.assert_series_equal(us["us_n_states_modis"], exp_m, check_names=False,
                                   check_dtype=False)
    pd.testing.assert_series_equal(us["us_n_states_weather"], exp_w, check_names=False,
                                   check_dtype=False)
    assert us["us_n_states_modis"].max() == 4 and us["us_n_states_weather"].min() < 5
    assert us["us_n_states_weather"].max() == 5
    # the aggregation itself is unchanged: still the renormalised weighted mean
    d = pd.Timestamp("2012-07-20")
    rows = p[p["date"] == d].set_index("geo")
    have = [s for s in STATES if pd.notna(rows.loc[s, "ndvi_anom"])]
    wt = F.us_weights(have)
    assert rows.loc["US", "ndvi_anom"] == pytest.approx(sum(wt[s] * rows.loc[s, "ndvi_anom"] for s in have))
    # carried into the trading table (US columns only, no per-state copies)
    zcz = pd.DataFrame({"date": pd.to_datetime(["2012-07-20"]), "contract": "ZCZ12",
                        "campaign": 2012, "close": 600.0, "ret": 0.0, "days_to_expiry": 100})
    t = F.build_trading(p, zcz)
    assert t.loc[0, "us_n_states_modis"] == 4 and t.loc[0, "us_n_states_weather"] == 4
    assert "us_n_states_modis_ia" not in t.columns


# --- trading table ------------------------------------------------------------------
@needs_real
def test_trading_table(real_panel):
    zcz = pd.read_parquet(DATA_PROCESSED / "zcz_front_dec.parquet")
    t = F.build_trading(real_panel, zcz)
    assert (t["campaign"] == t["date"].dt.year).all()
    assert t["date"].is_unique
    assert "ndvi_anom" in t and "ndvi_anom_ia" in t and "edd30_jul_ne" in t
    assert (t["modis_asof"].isna() | (t["modis_asof"] <= t["date"])).all()


# --- HLS (second satellite source) --------------------------------------------------
def _hls_build(inputs, seasons, hls, usda=False):
    return F.build_features(inputs["modis"], inputs["weather"],
                            inputs["usda"] if usda else None, seasons=seasons, hls=hls)


def test_sat_spec_names():
    assert F.MODIS_COLS == [
        "modis_age_days", "ndvi", "evi", "ndvi_anom", "evi_anom", "ndvi_z", "evi_z",
        "ndvi_anom_cum", "evi_anom_cum", "ndvi_trend_anom", "evi_trend_anom",
        "ndvi_cummax_anom", "modis_clim_n"]  # fmt: skip
    m = F.MODIS_SPEC
    assert (m.asof_col, m.coverage_col, m.n_states_col, m.anom_col) == (
        "modis_asof", "modis_coverage", "us_n_states_modis", "ndvi_anom")
    assert (m.lag_days, m.max_staleness_days, m.clim_first_year) == (14, 30, 2003)
    assert F.HLS_SPEC.block_cols == [
        "hls_asof", "hls_age_days", "hls_ndvi", "hls_evi", "hls_ndvi_anom", "hls_evi_anom",
        "hls_ndvi_z", "hls_evi_z", "hls_ndvi_anom_cum", "hls_evi_anom_cum",
        "hls_ndvi_trend_anom", "hls_evi_trend_anom", "hls_ndvi_cummax_anom", "hls_clim_n",
        "hls_coverage", "us_n_states_hls"]  # fmt: skip
    h = F.HLS_SPEC
    assert (h.lag_days, h.max_staleness_days, h.min_valid_frac, h.clim_first_year) == (
        F.HLS_LAG_DAYS, 21, 0.5, 2013)
    assert not h.has_good_cols and m.has_good_cols
    assert "us_n_states_hls" in F.US_ONLY_COLS


def test_hls_availability_overrides_stored(syn_hls_inputs):
    h = syn_hls_inputs["hls"]
    av = F.sat_available_date(h, F.HLS_SPEC)
    assert (av == h["end_date"] + pd.Timedelta(days=F.HLS_LAG_DAYS)).all()
    assert (av != h["available_date"]).all()  # stored end + 3 is ignored
    hp = F.prepare_sat(h, F.HLS_SPEC)
    assert (hp["valid_frac"] >= 0.5).all() and len(hp) < len(h)
    assert (hp["available_date"] == hp["end_date"] + pd.Timedelta(days=F.HLS_LAG_DAYS)).all()
    m = syn_hls_inputs["modis"]
    pd.testing.assert_series_equal(F.modis_available_date(m),
                                   F.sat_available_date(m, F.MODIS_SPEC))


def test_hls_leaves_modis_columns_identical(syn_hls_inputs, syn_hls_panel):
    i = syn_hls_inputs
    p0 = F.build_features(i["modis"], i["weather"], i["usda"], seasons=HLS_SYN_SEASONS)
    p = syn_hls_panel
    assert list(p.columns) == [*p0.columns, *F.HLS_SPEC.block_cols]  # appended last
    pd.testing.assert_frame_equal(p[list(p0.columns)], p0, check_exact=True)
    assert frame_digest(p[list(p0.columns)]) == frame_digest(p0)
    zcz = synth_zcz(HLS_SYN_SEASONS)
    t0, t = F.build_trading(p0, zcz), F.build_trading(p, zcz)
    n0 = len(t0.columns)
    assert list(t.columns[:n0]) == list(t0.columns)
    pd.testing.assert_frame_equal(t[list(t0.columns)], t0, check_exact=True)
    blk = F.HLS_SPEC.block_cols
    st_blk = [c for c in blk if c not in F.US_ONLY_COLS]
    assert list(t.columns[n0:]) == [*blk, *[f"{c}_{s.lower()}" for s in STATES for c in st_blk]]
    assert "us_n_states_hls_ia" not in t and "hls_ndvi_anom_ia" in t
    us = p[p["geo"] == "US"].set_index("date")
    np.testing.assert_array_equal(t["hls_ndvi_anom"].to_numpy(),
                                  us.loc[t["date"], "hls_ndvi_anom"].to_numpy())


def test_hls_asof_and_lag(syn_hls_inputs, syn_hls_panel):
    p = syn_hls_panel
    _check_asof(p)
    hp = F.prepare_sat(syn_hls_inputs["hls"], F.HLS_SPEC)
    ia = hp[(hp["state"] == "IA") & (hp["season"] == 2018)]
    g = p[(p["geo"] == "IA") & (p["season"] == 2018)].set_index("date")
    for d, row in g.iterrows():
        av = ia[ia["available_date"] <= d]
        if av.empty:
            assert pd.isna(row["hls_asof"]) and np.isnan(row["hls_age_days"])
            continue
        assert row["hls_asof"] == av["available_date"].max()
        latest = av.loc[av["t"].idxmax()]
        assert row["hls_age_days"] == (d - latest["end_date"]).days
        assert row["hls_age_days"] >= F.HLS_LAG_DAYS
        if row["hls_age_days"] <= F.HLS_MAX_STALENESS_DAYS:
            assert row["hls_ndvi"] == latest["ndvi_mean"]
    # a composite ending E is used from E + 7, never on E + 6
    c = ia[ia["end_date"].dt.month == 7].iloc[2]
    d = c["end_date"] + pd.Timedelta(days=F.HLS_LAG_DAYS)
    assert g.loc[d, "hls_ndvi"] == c["ndvi_mean"]
    assert g.loc[d - pd.Timedelta(days=1), "hls_ndvi"] != c["ndvi_mean"]


def test_hls_staleness(syn_hls_inputs):
    i = syn_hls_inputs
    h = i["hls"]
    h = h[~((h["year"] == 2018) & (h["start_date"] >= "2018-07-01"))]  # HLS outage
    p = _hls_build(i, [2018], h)
    hcols = [*F.HLS_VALUE_COLS]
    stale = p["hls_age_days"] > F.HLS_MAX_STALENESS_DAYS
    assert stale.any() and p.loc[stale, hcols].isna().all().all()
    hp = F.prepare_sat(h, F.HLS_SPEC)
    for st in STATES:
        last_end = hp.loc[(hp["state"] == st) & (hp["season"] == 2018), "end_date"].max()
        g = p[p["geo"] == st].set_index("date")
        ok = last_end + pd.Timedelta(days=F.HLS_MAX_STALENESS_DAYS)
        assert g.loc[ok, "hls_age_days"] == F.HLS_MAX_STALENESS_DAYS
        assert pd.notna(g.loc[ok, "hls_ndvi_anom"])
        bad = ok + pd.Timedelta(days=1)
        assert g.loc[bad:, hcols].isna().all().all()
        assert pd.notna(g.loc[bad, "hls_asof"])  # still known, just too old to use
        assert g.loc["2018-08-15":"2018-08-31", "ndvi_anom"].notna().all()  # MODIS unaffected
    us = p[p["geo"] == "US"].set_index("date").loc["2018-08-15":"2018-10-31"]
    assert us[hcols].isna().all().all()
    assert (us["us_n_states_hls"] == 0).all() and (us["hls_coverage"] == 0).all()
    assert (us["us_n_states_modis"] == 5).all()


def test_hls_climatology_prior_seasons_only(syn_hls_inputs):
    i = syn_hls_inputs
    h = i["hls"]
    hcols = F.HLS_SPEC.block_cols
    base = _hls_build(i, [2017, 2018], h)
    bumped = h.copy()
    bumped.loc[bumped["year"] >= 2018, ["ndvi_mean", "evi_mean"]] += 5.0
    after = _hls_build(i, [2017, 2018], bumped)
    b17, a17 = (x[x["season"] == 2017][hcols] for x in (base, after))
    pd.testing.assert_frame_equal(b17, a17, check_exact=True)  # 2018+ never reaches 2017
    b18, a18 = (x[x["season"] == 2018] for x in (base, after))
    pd.testing.assert_series_equal(b18["hls_clim_n"], a18["hls_clim_n"])
    diff = (a18["hls_ndvi_anom"] - b18["hls_ndvi_anom"]).dropna()
    assert len(diff) > 1000 and np.allclose(diff, 5.0)  # same climatology, own value moved
    cn = base.loc[base["geo"] != "US"].groupby("season")["hls_clim_n"].agg(["min", "max"])
    assert (cn.loc[2017] == 4).all() and (cn.loc[2018] == 5).all()  # 2013 .. Y-1
    # seasons before HLS_CLIM_FIRST_YEAR never enter (even when supplied)
    old = synth_hls([2011, 2012], seed=9, presample=True)
    old[["ndvi_mean", "evi_mean"]] += 3.0
    with_old = _hls_build(i, [2017, 2018], pd.concat([old, h], ignore_index=True))
    pd.testing.assert_frame_equal(with_old[hcols], base[hcols], check_exact=True)
    hp = F.prepare_sat(pd.concat([old, h]), F.HLS_SPEC)
    curves = F.season_curves(hp[hp["state"] == "IA"], "ndvi_mean")
    assert F.climatology_years(curves, 2016, F.HLS_SPEC.clim_first_year) == [2013, 2014, 2015]


def test_hls_presample_only_feeds_climatology(syn_hls_inputs, syn_hls_panel):
    i = syn_hls_inputs
    h = i["hls"]
    p = syn_hls_panel
    hcols = F.HLS_SPEC.block_cols
    s15 = p[p["season"] == 2015]
    assert s15[[*F.HLS_COLS]].isna().all().all() and s15["hls_asof"].isna().all()
    assert (s15["hls_coverage"] == 0).all()
    assert (s15.loc[s15["geo"] == "US", "us_n_states_hls"] == 0).all()
    s16 = p[(p["season"] == 2016) & (p["geo"] != "US")]
    assert (s16["hls_clim_n"].dropna() == 3).all()  # the 3 pre-sample seasons
    assert s16["hls_ndvi_anom"].notna().sum() > 1000
    # the flag is what keeps pre-sample rows out of their own season's rows
    unflagged = _hls_build(i, [2015, 2016], h.assign(presample=False))
    assert unflagged.loc[unflagged["season"] == 2015, "hls_ndvi"].notna().any()
    pd.testing.assert_frame_equal(
        unflagged.loc[unflagged["season"] == 2016, hcols].reset_index(drop=True),
        p.loc[p["season"] == 2016, hcols].reset_index(drop=True), check_exact=True)
    # without the pre-sample: same 2016 values, but no climatology yet
    nopre = _hls_build(i, [2016], h[~h["presample"]])
    ref = p[p["season"] == 2016].reset_index(drop=True)
    for c in ("hls_asof", "hls_age_days", "hls_ndvi", "hls_evi"):
        pd.testing.assert_series_equal(nopre[c], ref[c], check_exact=True)
    assert nopre["hls_ndvi_anom"].isna().all() and (nopre["hls_clim_n"].dropna() == 0).all()


def test_hls_trend_uses_composite_16d_earlier(syn_hls_panel):
    """With a regular 4-day cadence the trend partner is the composite ending 16 days
    before the latest one (non-overlapping window), i.e. the one that was the latest on
    d - 16 (so already available): trend(d) = anom(d) - anom(d - 16) whenever both rows
    use aligned composites (same age)."""
    lag = pd.Timedelta(days=F.HLS_SPEC.trend_lag_days)
    for geo in ("IA", "MN"):
        g = syn_hls_panel[(syn_hls_panel["geo"] == geo)
                          & (syn_hls_panel["season"] == 2019)].set_index("date")
        prev = g.reindex(g.index - lag).set_index(g.index)
        ok = ((g["hls_age_days"] == prev["hls_age_days"]) & g["hls_ndvi_anom"].notna()
              & prev["hls_ndvi_anom"].notna())
        assert ok.sum() > 100
        for v in ("ndvi", "evi"):
            exp = g.loc[ok, f"hls_{v}_anom"] - prev.loc[ok, f"hls_{v}_anom"]
            pd.testing.assert_series_equal(g.loc[ok, f"hls_{v}_trend_anom"], exp,
                                           check_names=False, check_exact=True)


def test_hls_us_aggregate(syn_hls_inputs):
    i = syn_hls_inputs
    h = i["hls"]
    p = _hls_build(i, [2018], h[h["state"] != "IL"])  # IL never has HLS
    us = p[p["geo"] == "US"].set_index("date")
    st = p[p["geo"] != "US"]
    exp_n = st.groupby("date")["hls_ndvi_anom"].apply(lambda s: s.notna().sum())
    pd.testing.assert_series_equal(us["us_n_states_hls"], exp_n, check_names=False,
                                   check_dtype=False)
    assert us["us_n_states_hls"].max() == 4
    assert set(st["hls_coverage"]) <= {0.0, 1.0}
    assert (st.loc[st["geo"] == "IL", "hls_coverage"] == 0).all()
    d = pd.Timestamp("2018-07-20")
    rows = p[p["date"] == d].set_index("geo")
    have = [s for s in STATES if pd.notna(rows.loc[s, "hls_ndvi_anom"])]
    assert len(have) == 4
    assert rows.loc["US", "hls_coverage"] == pytest.approx(1 - F.US_WEIGHTS["IL"])
    w = F.us_weights(have)
    assert rows.loc["US", "hls_ndvi_anom"] == pytest.approx(
        sum(w[s] * rows.loc[s, "hls_ndvi_anom"] for s in have))
    assert rows.loc["US", "hls_asof"] == rows.loc[list(STATES), "hls_asof"].max()
    assert rows.loc["US", "us_n_states_modis"] == 5  # MODIS composition unaffected


def test_load_hls_presample_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "HLS_STATE_PATH", tmp_path / "hls_state.parquet")
    monkeypatch.setattr(F, "HLS_PRESAMPLE_PATH", tmp_path / "hls_state_presample.parquet")
    assert F.load_hls() is None
    synth_hls([2016, 2017], states=["IA"]).drop(columns="presample").to_parquet(
        F.HLS_STATE_PATH)
    h = F.load_hls()
    assert not h["presample"].any() and set(h["year"]) == {2016, 2017}
    synth_hls([2014, 2015, 2016], states=["IA"]).drop(columns="presample").to_parquet(
        F.HLS_PRESAMPLE_PATH)
    h = F.load_hls()
    assert set(h.loc[h["presample"], "year"]) == {2014, 2015}  # 2016 is in the sample
    assert set(h.loc[~h["presample"], "year"]) == {2016, 2017}
    assert not F.load_hls(presample=False)["presample"].any()

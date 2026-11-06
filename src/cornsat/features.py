"""Phase 5: daily as-of feature panel (satellite, weather, optional USDA) with no look-ahead.

Output layout
-------------
``build_features`` returns a LONG panel: one row per (geo, date) for every calendar day
from SEASON_START (Apr 1) to SEASON_END (Oct 31) of each season, geo in the 5 core states
plus ``"US"``. A value on row (geo, d) is computed only from input rows whose
availability date is <= d; ``modis_asof`` / ``weather_asof`` / ``usda_asof`` give the
newest availability date in the information set used for that row (always <= d).
A row is "as of the morning of d": weather through d-WEATHER_LAG_DAYS (3), a MODIS
composite from ``end_date + MODIS_LAG_DAYS``, a NASS Monday report from Tuesday (see ``usda``).

Availability rules (overriding anything stored upstream)
---------------------------------------------------------
* MODIS: ``available = end_date + MODIS_LAG_DAYS`` (14 d; the stored 10 d is replaced
  because real latency was up to ~20 d for a few items). Composites with
  ``valid_frac < MIN_VALID_FRAC`` (0.5) are dropped.
* Weather: ``available_date = date + WEATHER_LAG_DAYS`` (3 d, as stored by ``weather``).
* USDA: ``available_date`` as stored by ``usda`` (Tuesday after the Sunday week end).

Climatology (the "normal" each anomaly is measured against)
-----------------------------------------------------------
For season Y the climatology uses only seasons CLIM_FIRST_YEAR (2003) .. Y-1, so it
expands each year and never contains season Y or later (``climatology``). Each prior
season's series is put on a daily day-of-year grid (``cal_doy``: a 365-day calendar,
leap days folded) -- MODIS and USDA condition by linear interpolation between
observations (no extrapolation beyond the season's first/last observation), weather by
the daily values themselves -- and the mean / std (ddof=1) across seasons is taken per
DOY. A DOY with fewer than MIN_CLIM_YEARS (3) seasons is NaN.

Pre-sample baseline: MODIS seasons 2003-2008 (``modis.build_presample``; Aqua starts
mid-2002) and weather 2003-2007 (``weather.build_weather_presample``) are kept in
separate files and used ONLY for climatology. CDL national coverage starts in 2008, so
the 2003-2008 MODIS seasons are corn-weighted with the CDL 2008 mask
(``cornmask.PRESAMPLE_CDL_YEARS``). This is an approximation: corn location in the
five states is very stable year to year and the baseline only normalises. All prior
seasons are fully available (last composite of Y-1 by mid-November of Y-1) before any
date of season Y, so the climatology is constant within a season. USDA tables cover
the sample years only, so USDA anomalies need 3 prior seasons (from 2012) unless older
NASS years are fetched. Without the pre-sample files the same code degrades to an
expanding window with a 3-season minimum (MODIS anomalies from 2012).

US aggregate
------------
US satellite/weather features = fixed-weight mean of the state features
(``weather.US_WEIGHTS``: IA .30, IL .26, NE .20, MN .14, IN .10), weights renormalised
over the states with a non-NaN value for that feature on that date (``us_weights``).
US z-scores are therefore weighted means of state z-scores. US USDA features use the
NASS national series (geo "US") directly. US ``*_asof`` = max of the state values.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from cornsat.config import DATA_PROCESSED, SEASON_END, SEASON_START, STATES, YEARS
from cornsat.weather import US_WEIGHTS, WEATHER_LAG_DAYS

log = logging.getLogger(__name__)

MODIS_LAG_DAYS = 14
MAX_STALENESS_DAYS = 30  # MODIS: NaN when (date - end_date of latest composite) > this
# NaN when the newest weather day is older than this: the normal lag (WEATHER_LAG_DAYS) plus
# 6 days of tolerated outage (the same 6 days of slack the former 1-day-lag / 7 d rule gave).
WEATHER_MAX_STALENESS_DAYS = WEATHER_LAG_DAYS + 6
USDA_MAX_STALENESS_DAYS = 14  # NaN when the newest NASS week_ending is older than this
US_ONLY_COLS = ("us_n_states_modis", "us_n_states_weather")  # defined on US rows only
MIN_VALID_FRAC = 0.5
CLIM_FIRST_YEAR = 2003
MIN_CLIM_YEARS = 3
TREND_LAG_DAYS = 16
TREND_TOL_DAYS = 6
CUM_START = (6, 1)
JULY = 7
EDD_SHORT_DAYS = 14
SILK_ZERO_UNTIL = (7, 1)

GEOS = [*STATES, "US"]
DOY_GRID = np.arange(1, 366)

FEATURES_DAILY_PATH = DATA_PROCESSED / "features_daily.parquet"
FEATURES_TRADING_PATH = DATA_PROCESSED / "features_trading.parquet"

MODIS_COLS = [
    "modis_age_days", "ndvi", "evi", "ndvi_anom", "evi_anom", "ndvi_z", "evi_z",
    "ndvi_anom_cum", "evi_anom_cum", "ndvi_trend_anom", "evi_trend_anom",
    "ndvi_cummax_anom", "modis_clim_n",
]  # fmt: skip
MODIS_VALUE_COLS = [c for c in MODIS_COLS if c not in ("modis_age_days", "modis_clim_n")]
WEATHER_COLS = [
    "gdd_cum", "gdd_cum_anom", "precip_cum", "precip_cum_anom", "precip_30d",
    "precip_30d_anom", "edd30_cum", "edd30_jul", "edd30_14d",
]  # fmt: skip
USDA_COLS = ["ge", "ci", "ge_anom", "ge_chg_2w", "pct_silking", "silking_anom"]


# --- calendar helpers ---------------------------------------------------------------
def cal_doy(dates) -> np.ndarray:
    """Day of year on a 365-day calendar: in leap years days after Feb 28 shift by -1, so
    the same calendar day has the same DOY every year (Feb 29 shares DOY 60 with Mar 1)."""
    d = pd.DatetimeIndex(pd.to_datetime(dates))
    shift = (d.is_leap_year & (d.month > 2)).astype(int)
    return np.asarray(d.day_of_year - shift, dtype="int64")


def _dt(x):
    """Datetime column/index normalised to datetime64[ns] (merge keys must match)."""
    return pd.to_datetime(x).astype("datetime64[ns]")


def season_days(season: int) -> pd.DatetimeIndex:
    return _dt(pd.date_range(date(season, *SEASON_START), date(season, *SEASON_END), freq="D"))


def us_weights(states) -> dict[str, float]:
    """US_WEIGHTS renormalised over ``states`` (those with data); sums to 1."""
    w = {s: US_WEIGHTS[s] for s in states if s in US_WEIGHTS}
    tot = sum(w.values())
    if tot <= 0:
        return {}
    return {s: v / tot for s, v in w.items()}


# --- climatology --------------------------------------------------------------------
def season_curves(df: pd.DataFrame, value: str, doy: str = "doy", season: str = "season",
                  fill_left=np.nan, fill_right=np.nan, grid=DOY_GRID) -> pd.DataFrame:
    """One row per season, one column per DOY of ``grid``: each season's observations of
    ``value`` linearly interpolated to the daily grid. Outside a season's observed range
    the value is ``fill_left`` / ``fill_right`` (NaN = no extrapolation; the string
    "last" carries the season's last value). Duplicate DOYs are averaged."""
    rows = {}
    for s, g in df[[season, doy, value]].dropna().groupby(season):
        g = g.groupby(doy)[value].mean().sort_index()
        x, y = g.index.to_numpy(float), g.to_numpy(float)
        right = y[-1] if isinstance(fill_right, str) and fill_right == "last" else fill_right
        rows[s] = np.interp(grid, x, y, left=fill_left, right=right)
    out = pd.DataFrame.from_dict(rows, orient="index", columns=grid).astype("float64")
    out.index.name = season
    return out.sort_index()


def climatology(curves: pd.DataFrame, season: int, first_year: int = CLIM_FIRST_YEAR,
                min_years: int = MIN_CLIM_YEARS) -> pd.DataFrame:
    """Per-DOY mean, std (ddof=1) and n over seasons first_year .. season-1 ONLY.

    Returns a frame indexed by DOY with columns mean, std, n (mean/std NaN where n <
    min_years). Season ``season`` and later are never used (no look-ahead)."""
    prior = curves.loc[(curves.index >= first_year) & (curves.index < season)]
    n = prior.notna().sum(axis=0)
    ok = n >= min_years
    return pd.DataFrame({
        "mean": prior.mean(axis=0).where(ok),
        "std": prior.std(axis=0, ddof=1).where(ok),
        "n": n.astype(int),
    })


def climatology_years(curves: pd.DataFrame, season: int,
                      first_year: int = CLIM_FIRST_YEAR) -> list[int]:
    """Seasons that ``climatology(curves, season)`` draws on."""
    idx = curves.index
    return [int(y) for y in idx[(idx >= first_year) & (idx < season)]]


# --- input preparation --------------------------------------------------------------
def modis_available_date(df: pd.DataFrame) -> pd.Series:
    """Availability used by Phase 5: end_date + MODIS_LAG_DAYS (overrides stored value)."""
    return _dt(df["end_date"]) + pd.Timedelta(days=MODIS_LAG_DAYS)


def prepare_modis(modis: pd.DataFrame) -> pd.DataFrame:
    """Valid composites with Phase-5 availability, observation time ``t`` (obs_date, or
    mid_date if missing), ``season`` and ``doy``. Terra and Aqua are merged into one
    series per state (8-day cadence) simply by pooling and ordering on ``t``."""
    df = modis[modis["valid_frac"] >= MIN_VALID_FRAC].copy()
    for c in ["start_date", "end_date", "mid_date", "obs_date"]:
        df[c] = _dt(df[c])
    df["available_date"] = modis_available_date(df)
    df["t"] = df["obs_date"].fillna(df["mid_date"])
    df["season"] = df["year"].astype(int)
    df["doy"] = cal_doy(df["t"])
    df = df.dropna(subset=["ndvi_mean", "evi_mean"])
    return df.sort_values(["state", "available_date", "t"]).reset_index(drop=True)


def prepare_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Per-state daily weather (US rows dropped: the US aggregate is formed from the state
    features), with ``season``, ``doy``, ``edd30_jul`` (EDD summed from Jul 1 through the
    row's date, 0 before July, July total after) and ``edd30_14d`` (EDD over the 14 days
    ending on the row's date). Both use only rows at or before the row (causal)."""
    w = weather[weather["state"].isin(list(STATES))].copy()
    w["date"] = _dt(w["date"])
    w["available_date"] = _dt(w["available_date"])
    w = w.drop_duplicates(["state", "date"], keep="last").sort_values(["state", "date"])
    w["season"] = w["date"].dt.year
    w["doy"] = cal_doy(w["date"])
    jul = w["edd30"].where(w["date"].dt.month == JULY, 0.0)
    w["edd30_jul"] = jul.groupby([w["state"], w["season"]]).cumsum()
    w["edd30_14d"] = w.groupby("state")["edd30"].transform(
        lambda s: s.rolling(EDD_SHORT_DAYS, min_periods=EDD_SHORT_DAYS).sum())
    # a gap in the daily series would make a 14-row window span > 14 days
    span = w.groupby("state")["date"].diff(EDD_SHORT_DAYS - 1).dt.days
    w.loc[span != EDD_SHORT_DAYS - 1, "edd30_14d"] = np.nan
    return w.reset_index(drop=True)


def prepare_usda(usda: dict | None) -> dict | None:
    """Condition and progress tables with ``season``/``doy``. Silking (``silk``) is filled
    causally within each (geo, season): after the first report, the last reported value
    (NASS stops reporting near 100 %); before the first report, 0 if the week ends
    before SILK_ZERO_UNTIL (Jul 1; silking has not started), else NaN (missing data)."""
    if not usda:
        return None
    out = {}
    cond = usda.get("condition")
    if cond is not None and len(cond):
        c = cond.copy()
        c["week_ending"] = _dt(c["week_ending"])
        c["available_date"] = _dt(c["available_date"])
        c["season"] = c["week_ending"].dt.year
        c["doy"] = cal_doy(c["week_ending"])
        out["condition"] = c.sort_values(["geo", "week_ending"]).reset_index(drop=True)
    prog = usda.get("progress")
    if prog is not None and len(prog) and "pct_silking" in prog:
        p = prog.copy()
        p["week_ending"] = _dt(p["week_ending"])
        p["available_date"] = _dt(p["available_date"])
        p["season"] = p["week_ending"].dt.year
        p["doy"] = cal_doy(p["week_ending"])
        p = p.sort_values(["geo", "week_ending"]).reset_index(drop=True)
        key = [p["geo"], p["season"]]
        seen = p["pct_silking"].notna().astype(int).groupby(key).cummax().astype(bool)
        early = p["week_ending"].dt.month * 100 + p["week_ending"].dt.day < (
            SILK_ZERO_UNTIL[0] * 100 + SILK_ZERO_UNTIL[1])
        pre = np.where(early, 0.0, np.nan)
        p["silk"] = p.groupby(["geo", "season"])["pct_silking"].ffill().where(seen, pre)
        out["progress"] = p
    return out or None


# --- MODIS --------------------------------------------------------------------------
def _modis_state(comps: pd.DataFrame, seasons) -> pd.DataFrame:
    """Daily MODIS features for one state (comps = prepared rows of that state)."""
    curves = {v: season_curves(comps, f"{v}_mean") for v in ("ndvi", "evi")}
    apr1_doy = int(cal_doy([pd.Timestamp(2001, *SEASON_START)])[0])
    frames = []
    for y in seasons:
        days = season_days(y)
        out = pd.DataFrame(index=days, columns=[*MODIS_COLS, "modis_asof"], dtype="float64")
        out["modis_asof"] = pd.NaT
        cs = comps[comps["season"] == y].sort_values(["available_date", "t"])
        if cs.empty:
            frames.append(out)
            continue
        clim = {v: climatology(curves[v], y) for v in ("ndvi", "evi")}
        doy = cs["doy"].to_numpy()
        val = {v: cs[f"{v}_mean"].to_numpy(float) for v in ("ndvi", "evi")}
        anom = {v: val[v] - clim[v]["mean"].reindex(doy).to_numpy() for v in val}
        z = {v: anom[v] / clim[v]["std"].reindex(doy).to_numpy() for v in val}
        t = cs["t"].to_numpy("datetime64[D]")
        end = cs["end_date"].to_numpy("datetime64[D]")
        avail = cs["available_date"].to_numpy("datetime64[D]")
        cum_start = np.datetime64(date(y, *CUM_START))
        apr1 = np.datetime64(date(y, *SEASON_START))
        clim_n = clim["ndvi"]["n"].reindex(doy).to_numpy()
        ndvi_clim_mean = clim["ndvi"]["mean"]

        prefix_rows = []
        for k in range(1, len(cs) + 1):
            tt = t[:k]
            li = int(np.flatnonzero(tt == tt.max())[-1])  # latest observation in the prefix
            r = {"end": end[li], "asof": avail[:k].max(), "clim_n": clim_n[li]}
            for v in ("ndvi", "evi"):
                r[v] = val[v][li]
                r[f"{v}_anom"] = anom[v][li]
                r[f"{v}_z"] = z[v][li]
                m = tt >= cum_start
                r[f"{v}_anom_cum"] = np.nanmean(anom[v][:k][m]) if np.isfinite(
                    anom[v][:k][m]).any() else np.nan
            target = tt[li] - np.timedelta64(TREND_LAG_DAYS, "D")
            dist = np.abs((tt - target).astype(int)).astype(float)
            dist[li] = np.inf
            dist[dist > TREND_TOL_DAYS] = np.inf
            if np.isfinite(dist).any():
                pj = int(np.flatnonzero(dist == dist.min())[-1])
                for v in ("ndvi", "evi"):
                    r[f"{v}_trend_anom"] = anom[v][li] - anom[v][pj]
            else:
                r["ndvi_trend_anom"] = r["evi_trend_anom"] = np.nan
            m = tt >= apr1
            seg = ndvi_clim_mean.loc[apr1_doy: doy[li]].to_numpy()
            if m.any() and np.isfinite(seg).any() and tt[li] >= apr1:
                r["ndvi_cummax_anom"] = val["ndvi"][:k][m].max() - np.nanmax(seg)
            else:
                r["ndvi_cummax_anom"] = np.nan
            prefix_rows.append(r)
        pr = pd.DataFrame(prefix_rows)

        k = np.searchsorted(avail, days.to_numpy("datetime64[D]"), side="right")
        has = k > 0
        sel = pr.iloc[k[has] - 1].reset_index(drop=True)
        d_has = days[has]
        age = (d_has.to_numpy("datetime64[D]") - sel["end"].to_numpy("datetime64[D]")
               ).astype(int)
        fresh = age <= MAX_STALENESS_DAYS
        out.loc[d_has, "modis_age_days"] = age.astype(float)
        out.loc[d_has, "modis_asof"] = pd.to_datetime(sel["asof"]).to_numpy()
        out.loc[d_has, "modis_clim_n"] = sel["clim_n"].to_numpy(float)
        for c in MODIS_VALUE_COLS:
            out.loc[d_has, c] = np.where(fresh, sel[c].to_numpy(float), np.nan)
        frames.append(out)
    res = pd.concat(frames)
    res.index.name = "date"
    res["modis_asof"] = pd.to_datetime(res["modis_asof"])
    return res


# --- weather ------------------------------------------------------------------------
def _weather_state(w: pd.DataFrame, seasons) -> pd.DataFrame:
    """Daily weather features for one state (w = prepared rows of that state)."""
    anom_cols = ("gdd_cum", "precip_cum", "precip_30d")
    pivots = {c: w.pivot_table(index="season", columns="doy", values=c, aggfunc="mean")
              .reindex(columns=DOY_GRID) for c in anom_cols}
    w = w.sort_values("available_date")
    frames = []
    for y in seasons:
        days = pd.DataFrame({"date": season_days(y)})
        wr = w.drop(columns=["season"]).rename(columns={"date": "wx_date"})
        m = pd.merge_asof(days, wr, left_on="date", right_on="available_date",
                          direction="backward")
        out = pd.DataFrame(index=days["date"])
        out["weather_asof"] = m["available_date"].to_numpy()
        age = (days["date"] - pd.to_datetime(m["wx_date"])).dt.days
        stale = (age.isna() | (age > WEATHER_MAX_STALENESS_DAYS)).to_numpy()
        for c in ["gdd_cum", "precip_cum", "precip_30d", "edd30_cum", "edd30_jul", "edd30_14d"]:
            out[c] = np.where(stale, np.nan, m[c].to_numpy(float))
        doy = m["doy"].to_numpy()
        for c in anom_cols:
            clim = climatology(pivots[c], y)["mean"]
            ref = clim.reindex(np.nan_to_num(doy, nan=0).astype(int)).to_numpy()
            out[f"{c}_anom"] = out[c].to_numpy() - ref
        frames.append(out)
    res = pd.concat(frames)
    res.index.name = "date"
    res["weather_asof"] = pd.to_datetime(res["weather_asof"])
    return res


# --- USDA ---------------------------------------------------------------------------
def _usda_geo(u: dict, geo: str, seasons) -> pd.DataFrame | None:
    cond = u.get("condition")
    prog = u.get("progress")
    cond = cond[cond["geo"] == geo] if cond is not None else None
    prog = prog[prog["geo"] == geo] if prog is not None else None
    if (cond is None or cond.empty) and (prog is None or prog.empty):
        return None
    ge_curves = season_curves(cond, "ge") if cond is not None and len(cond) else None
    silk_curves = (season_curves(prog, "silk", fill_left=0.0, fill_right="last")
                   if prog is not None and len(prog) else None)
    frames = []
    for y in seasons:
        days = season_days(y)
        dd = days.to_numpy("datetime64[D]")
        out = pd.DataFrame(index=days, columns=USDA_COLS, dtype="float64")
        asof = pd.Series(pd.NaT, index=days, dtype="datetime64[ns]")
        if cond is not None:
            cs = cond[cond["season"] == y].sort_values("available_date")
            if len(cs):
                k = np.searchsorted(cs["available_date"].to_numpy("datetime64[D]"), dd, "right")
                has = k > 0
                sel = cs.iloc[k[has] - 1]
                we = sel["week_ending"].to_numpy("datetime64[D]")
                fresh = (dd[has] - we).astype(int) <= USDA_MAX_STALENESS_DAYS
                clim = climatology(ge_curves, y)["mean"]
                ge = sel["ge"].to_numpy(float)
                ge_map = dict(zip(cs["week_ending"].to_numpy("datetime64[D]"),
                                  cs["ge"].to_numpy(float)))
                prev = np.array([ge_map.get(x - np.timedelta64(14, "D"), np.nan) for x in we])
                vals = {
                    "ge": ge, "ci": sel["ci"].to_numpy(float),
                    "ge_anom": ge - clim.reindex(sel["doy"].to_numpy()).to_numpy(),
                    "ge_chg_2w": ge - prev,
                }
                for c, v in vals.items():
                    out.loc[days[has], c] = np.where(fresh, v, np.nan)
                asof[days[has]] = sel["available_date"].to_numpy()
        if prog is not None:
            ps = prog[prog["season"] == y].sort_values("available_date")
            if len(ps):
                k = np.searchsorted(ps["available_date"].to_numpy("datetime64[D]"), dd, "right")
                has = k > 0
                sel = ps.iloc[k[has] - 1]
                we = sel["week_ending"].to_numpy("datetime64[D]")
                fresh = (dd[has] - we).astype(int) <= USDA_MAX_STALENESS_DAYS
                clim = climatology(silk_curves, y)["mean"]
                silk = sel["silk"].to_numpy(float)
                out.loc[days[has], "pct_silking"] = np.where(fresh, silk, np.nan)
                out.loc[days[has], "silking_anom"] = np.where(
                    fresh, silk - clim.reindex(sel["doy"].to_numpy()).to_numpy(), np.nan)
                pa = pd.Series(sel["available_date"].to_numpy(), index=days[has])
                asof[days[has]] = np.maximum(asof[days[has]].fillna(pa), pa)
        out["usda_asof"] = asof
        frames.append(out)
    res = pd.concat(frames)
    res.index.name = "date"
    return res


# --- panel --------------------------------------------------------------------------
def _us_aggregate(states_panel: pd.DataFrame, value_cols, asof_cols) -> pd.DataFrame:
    """US rows = renormalised fixed-weight mean of state values (per date and column)."""
    p = states_panel
    w = p["geo"].map(US_WEIGHTS).astype(float)
    out = {}
    for c in value_cols:
        x = p[c].astype(float)
        ok = x.notna()
        num = (x.where(ok, 0.0) * w).groupby(p["date"]).sum()
        den = w.where(ok, 0.0).groupby(p["date"]).sum()
        out[c] = (num / den).where(den > 0)
    for c in asof_cols:
        out[c] = p.groupby("date")[c].max()
    res = pd.DataFrame(out)
    res["modis_coverage"] = (w.where(p["ndvi_anom"].notna(), 0.0)
                             .groupby(p["date"]).sum() / sum(US_WEIGHTS.values()))
    # number of states entering the aggregate (renormalised weights: fewer than 5 means the
    # aggregate is a sub-national mean); MODIS counted on ndvi_anom, weather on gdd_cum_anom
    res["us_n_states_modis"] = p["ndvi_anom"].notna().groupby(p["date"]).sum().astype(int)
    res["us_n_states_weather"] = p["gdd_cum_anom"].notna().groupby(p["date"]).sum().astype(int)
    return res.reset_index()


def build_features(modis: pd.DataFrame, weather: pd.DataFrame, usda: dict | None = None,
                   seasons=None) -> pd.DataFrame:
    """Daily as-of panel (long; see module doc) from PRE-LOADED input frames.

    ``modis``: modis_state rows (sample and pre-sample seasons concatenated; the stored
    available_date is ignored and replaced by end_date + MODIS_LAG_DAYS). ``weather``:
    weather_daily rows (sample + pre-sample). ``usda``: optional dict with
    ``condition`` / ``progress`` frames as produced by ``usda.parse_*``. ``seasons``:
    output seasons (default config.YEARS). Every input row is used only on dates >= its
    availability date, so passing frames truncated at date d yields identical values
    for all rows dated <= d (the truncation test relies on this).
    """
    seasons = list(YEARS if seasons is None else seasons)
    mp = prepare_modis(modis)
    wp = prepare_weather(weather)
    up = prepare_usda(usda)

    parts = []
    for st in STATES:
        f_m = _modis_state(mp[mp["state"] == st], seasons)
        f_w = _weather_state(wp[wp["state"] == st], seasons)
        f = f_m.join(f_w, how="outer")
        f.insert(0, "geo", st)
        parts.append(f.reset_index())
    states_panel = pd.concat(parts, ignore_index=True)
    us = _us_aggregate(states_panel, [*MODIS_COLS, *WEATHER_COLS],
                       ["modis_asof", "weather_asof"])
    us.insert(0, "geo", "US")
    panel = pd.concat([states_panel, us], ignore_index=True)
    panel.loc[panel["geo"] != "US", "modis_coverage"] = np.where(
        panel.loc[panel["geo"] != "US", "ndvi_anom"].notna(), 1.0, 0.0)

    if up is not None:
        uparts = []
        for g in GEOS:
            f = _usda_geo(up, g, seasons)
            if f is not None:
                f.insert(0, "geo", g)
                uparts.append(f.reset_index())
        if uparts:
            panel = panel.merge(pd.concat(uparts, ignore_index=True), on=["geo", "date"],
                                how="left")

    panel["season"] = panel["date"].dt.year
    panel["doy"] = cal_doy(panel["date"])
    lead = ["geo", "date", "season", "doy", "modis_asof", "weather_asof"]
    if "usda_asof" in panel:
        lead.append("usda_asof")
    rest = [c for c in panel.columns if c not in lead]
    panel = panel[lead + rest]
    panel["geo"] = pd.Categorical(panel["geo"], categories=GEOS)
    panel = panel.sort_values(["geo", "date"]).reset_index(drop=True)
    panel["geo"] = panel["geo"].astype(str)
    return panel


# --- loading / outputs --------------------------------------------------------------
def load_usda_tables() -> dict | None:
    """USDA condition/progress/yield parquet files if present, else None."""
    out = {}
    for name in ("condition", "progress", "yield"):
        p = DATA_PROCESSED / f"usda_{name}.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
    return out or None


def load_inputs(presample: bool = True) -> dict:
    """Pre-loaded inputs for ``build_features``: modis (with ``presample`` flag column),
    weather (sample + pre-sample) and usda (dict or None)."""
    from cornsat import modis as modis_mod
    from cornsat import weather as weather_mod

    m = modis_mod.load_modis_state().assign(presample=False)
    w = pd.read_parquet(DATA_PROCESSED / "weather_daily.parquet")
    if presample:
        mp = modis_mod.load_modis_presample()
        if mp is not None:
            m = pd.concat([mp.assign(presample=True), m], ignore_index=True)
        else:
            log.warning("no MODIS pre-sample; climatology falls back to expanding, min %d",
                        MIN_CLIM_YEARS)
        wp = weather_mod.load_weather_presample()
        if wp is not None:
            w = pd.concat([wp, w], ignore_index=True)
        else:
            log.warning("no weather pre-sample; climatology falls back to expanding window")
    return {"modis": m, "weather": w, "usda": load_usda_tables()}


def build_trading(panel: pd.DataFrame, zcz: pd.DataFrame) -> pd.DataFrame:
    """US features (unsuffixed) plus every state's features (suffix ``_<state>``, lower
    case) joined onto front-December ZCZ trading dates that fall in the season window of
    their own campaign (Apr 1 - Oct 31 of year == campaign). Features on row d are as of
    d (see module doc)."""
    z = zcz[["date", "contract", "campaign", "close", "ret", "days_to_expiry"]].copy()
    z["date"] = _dt(z["date"])
    z = z[z["campaign"] == z["date"].dt.year]
    feat_cols = [c for c in panel.columns if c not in ("geo", "date", "season", "doy")]
    us = panel[panel["geo"] == "US"][["date", *feat_cols]]
    out = z.merge(us, on="date", how="inner")
    st_cols = [c for c in feat_cols if c not in US_ONLY_COLS]
    for st in STATES:
        s = panel[panel["geo"] == st][["date", *st_cols]]
        s = s.rename(columns={c: f"{c}_{st.lower()}" for c in st_cols})
        out = out.merge(s, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def build_and_save(presample: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    inp = load_inputs(presample)
    panel = build_features(inp["modis"], inp["weather"], inp["usda"])
    zcz = pd.read_parquet(DATA_PROCESSED / "zcz_front_dec.parquet")
    trading = build_trading(panel, zcz)
    FEATURES_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FEATURES_DAILY_PATH, index=False)
    trading.to_parquet(FEATURES_TRADING_PATH, index=False)
    return panel, trading

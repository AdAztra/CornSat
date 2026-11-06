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

Satellite sources (``SatSpec``)
-------------------------------
MODIS and NASA HLS (30 m, ``hls_state.parquet``) go through the same recipe
(``prepare_sat`` / ``_sat_state``); a ``SatSpec`` carries what differs. ``MODIS_SPEC``
keeps the historical unprefixed names (ndvi, ..., modis_asof, modis_coverage,
us_n_states_modis). ``HLS_SPEC`` prefixes every column with ``hls_`` (hls_ndvi, ...,
hls_asof, hls_coverage, us_n_states_hls); HLS columns are appended AFTER all other
columns, so a build without HLS (``hls=None``) is bit-identical to the MODIS-only panel.
HLS composites end every 4 days over a 16-day trailing window (they overlap): available
= ``end_date + HLS_LAG_DAYS`` (7 d), stale after ``HLS_MAX_STALENESS_DAYS`` (21 d) from
the composite's end_date, climatology from HLS_CLIM_FIRST_YEAR (2013) .. Y-1 with the
2013-2015 pre-sample (``hls_state_presample.parquet``) feeding only the climatology.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from cornsat.config import DATA_PROCESSED, SEASON_END, SEASON_START, STATES, YEARS
from cornsat.weather import US_WEIGHTS, WEATHER_LAG_DAYS

log = logging.getLogger(__name__)

MODIS_LAG_DAYS = 14
MAX_STALENESS_DAYS = 30  # MODIS: NaN when (date - end_date of latest composite) > this
# HLS (30 m, 4-day cadence of 16-day trailing composites). The lag is provisional (to be
# re-measured from real HLS ingestion latency); staleness tolerates ~3 missed composites.
HLS_LAG_DAYS = 7
HLS_MAX_STALENESS_DAYS = 21
HLS_MIN_VALID_FRAC = 0.5
HLS_CLIM_FIRST_YEAR = 2013  # Landsat-8 L30 starts in 2013; S30 from late 2015
# NaN when the newest weather day is older than this: the normal lag (WEATHER_LAG_DAYS) plus
# 6 days of tolerated outage (the same 6 days of slack the former 1-day-lag / 7 d rule gave).
WEATHER_MAX_STALENESS_DAYS = WEATHER_LAG_DAYS + 6
USDA_MAX_STALENESS_DAYS = 14  # NaN when the newest NASS week_ending is older than this
# defined on US rows only (not copied per state into the trading table)
US_ONLY_COLS = ("us_n_states_modis", "us_n_states_weather", "us_n_states_hls")
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
HLS_STATE_PATH = DATA_PROCESSED / "hls_state.parquet"
HLS_PRESAMPLE_PATH = DATA_PROCESSED / "hls_state_presample.parquet"

# value features of every satellite source, unprefixed (MODIS keeps exactly these names)
SAT_VALUE_BASE = [
    "ndvi", "evi", "ndvi_anom", "evi_anom", "ndvi_z", "evi_z", "ndvi_anom_cum",
    "evi_anom_cum", "ndvi_trend_anom", "evi_trend_anom", "ndvi_cummax_anom",
]  # fmt: skip


@dataclass(frozen=True)
class SatSpec:
    """One satellite source of the panel: column naming and timing rules.

    ``name`` names the meta columns (``<name>_age_days``, ``<name>_clim_n``,
    ``<name>_asof``, ``<name>_coverage``, ``us_n_states_<name>``); ``prefix`` is put in
    front of the value columns (``SAT_VALUE_BASE``). A composite is available on
    ``end_date + lag_days``; values are NaN when ``date - end_date`` of the composite used
    exceeds ``max_staleness_days``; composites with ``valid_frac < min_valid_frac`` are
    dropped; the climatology uses seasons ``clim_first_year`` .. Y-1 with at least
    ``min_clim_years`` per DOY. ``has_good_cols`` records whether the input carries
    QA-good-only ``*_mean_good`` columns (MODIS); the features use ``*_mean`` either way.
    """

    name: str
    prefix: str
    lag_days: int
    max_staleness_days: int
    min_valid_frac: float
    clim_first_year: int
    has_good_cols: bool
    min_clim_years: int = MIN_CLIM_YEARS
    trend_lag_days: int = TREND_LAG_DAYS
    trend_tol_days: int = TREND_TOL_DAYS

    @property
    def value_cols(self) -> list[str]:
        return [self.prefix + c for c in SAT_VALUE_BASE]

    @property
    def age_col(self) -> str:
        return f"{self.name}_age_days"

    @property
    def clim_n_col(self) -> str:
        return f"{self.name}_clim_n"

    @property
    def asof_col(self) -> str:
        return f"{self.name}_asof"

    @property
    def coverage_col(self) -> str:
        return f"{self.name}_coverage"

    @property
    def n_states_col(self) -> str:
        return f"us_n_states_{self.name}"

    @property
    def anom_col(self) -> str:
        """Column that defines coverage / n_states (a state "has" the source that day)."""
        return f"{self.prefix}ndvi_anom"

    @property
    def cols(self) -> list[str]:
        """Per-state numeric columns (also US-aggregated by weighted mean)."""
        return [self.age_col, *self.value_cols, self.clim_n_col]

    @property
    def block_cols(self) -> list[str]:
        """Every panel column this source produces, in the order an appended block uses."""
        return [self.asof_col, *self.cols, self.coverage_col, self.n_states_col]


MODIS_SPEC = SatSpec("modis", "", MODIS_LAG_DAYS, MAX_STALENESS_DAYS, MIN_VALID_FRAC,
                     CLIM_FIRST_YEAR, has_good_cols=True)
HLS_SPEC = SatSpec("hls", "hls_", HLS_LAG_DAYS, HLS_MAX_STALENESS_DAYS, HLS_MIN_VALID_FRAC,
                   HLS_CLIM_FIRST_YEAR, has_good_cols=False)
SAT_SPECS = (MODIS_SPEC, HLS_SPEC)

MODIS_COLS = MODIS_SPEC.cols
MODIS_VALUE_COLS = MODIS_SPEC.value_cols
HLS_COLS = HLS_SPEC.cols
HLS_VALUE_COLS = HLS_SPEC.value_cols
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
def sat_available_date(df: pd.DataFrame, spec: SatSpec) -> pd.Series:
    """Availability used by the panel: end_date + spec.lag_days (overrides any stored
    available_date). For overlapping HLS composites each composite has its own end_date, so
    the composite ending E is usable from E + lag even though later composites share part
    of its window (they become usable only from their own, later, end_date + lag)."""
    return _dt(df["end_date"]) + pd.Timedelta(days=spec.lag_days)


def modis_available_date(df: pd.DataFrame) -> pd.Series:
    """Availability used by Phase 5: end_date + MODIS_LAG_DAYS (overrides stored value)."""
    return sat_available_date(df, MODIS_SPEC)


def prepare_sat(sat: pd.DataFrame, spec: SatSpec) -> pd.DataFrame:
    """Valid composites with panel availability, observation time ``t`` (obs_date, or
    mid_date if missing), ``season``, ``doy`` and a boolean ``presample`` (False when the
    column is absent). Several platforms (MODIS Terra + Aqua, 8-day cadence; HLS is a
    single harmonised L30+S30 series) are merged into one series per state simply by
    pooling and ordering on ``t``."""
    df = sat[sat["valid_frac"] >= spec.min_valid_frac].copy()
    for c in ["start_date", "end_date", "mid_date", "obs_date"]:
        df[c] = _dt(df[c])
    df["available_date"] = sat_available_date(df, spec)
    df["t"] = df["obs_date"].fillna(df["mid_date"])
    df["season"] = df["year"].astype(int)
    df["doy"] = cal_doy(df["t"])
    df["presample"] = (df["presample"].fillna(False).astype(bool) if "presample" in df
                       else False)
    df = df.dropna(subset=["ndvi_mean", "evi_mean"])
    return df.sort_values(["state", "available_date", "t"]).reset_index(drop=True)


def prepare_modis(modis: pd.DataFrame) -> pd.DataFrame:
    """``prepare_sat`` with ``MODIS_SPEC`` (Terra + Aqua pooled, 8-day cadence)."""
    return prepare_sat(modis, MODIS_SPEC)


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


# --- satellite (MODIS, HLS) ---------------------------------------------------------
def _sat_state(comps: pd.DataFrame, seasons, spec: SatSpec) -> pd.DataFrame:
    """Daily features of one satellite source for one state (comps = ``prepare_sat`` rows
    of that state), columns ``[*spec.cols, spec.asof_col]``.

    No look-ahead, also with overlapping composites (HLS: 16-day windows every 4 days):
    * Row d uses only composites with ``available_date = end_date + lag <= d``: the season's
      composites are ordered by availability, features are computed once per available
      prefix and row d takes the prefix ``searchsorted(avail, d, "right")``.
    * Values come from the composite with the latest observation time ``t`` in that
      prefix; staleness (``<name>_age_days``) is measured from that composite's end_date,
      the last day its window can contain.
    * The trend partner is the composite of the SAME prefix whose ``t`` is nearest to
      t_latest - trend_lag_days (16 d; within trend_tol_days), so it was already available.
      For HLS at a 4-day cadence that is normally the composite ending 16 days earlier,
      whose window does not overlap the latest one.
    * The climatology uses seasons clim_first_year .. Y-1 only (all available before the
      Apr 1 start of season Y). ``presample`` rows enter the climatology curves but never
      the features of their own season (pre-sample seasons give NaN rows).
    * ``*_anom_cum`` averages overlapping composites (each day counted in up to 4 HLS
      composites): a smoothing, not a look-ahead.
    """
    curves = {v: season_curves(comps, f"{v}_mean") for v in ("ndvi", "evi")}
    apr1_doy = int(cal_doy([pd.Timestamp(2001, *SEASON_START)])[0])
    own = ~comps["presample"].to_numpy(bool) if "presample" in comps else True
    frames = []
    for y in seasons:
        days = season_days(y)
        out = pd.DataFrame(index=days, columns=[*spec.cols, spec.asof_col], dtype="float64")
        out[spec.asof_col] = pd.NaT
        cs = comps[(comps["season"] == y) & own].sort_values(["available_date", "t"])
        if cs.empty:
            frames.append(out)
            continue
        clim = {v: climatology(curves[v], y, spec.clim_first_year, spec.min_clim_years)
                for v in ("ndvi", "evi")}
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
            target = tt[li] - np.timedelta64(spec.trend_lag_days, "D")
            dist = np.abs((tt - target).astype(int)).astype(float)
            dist[li] = np.inf
            dist[dist > spec.trend_tol_days] = np.inf
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
        fresh = age <= spec.max_staleness_days
        out.loc[d_has, spec.age_col] = age.astype(float)
        out.loc[d_has, spec.asof_col] = pd.to_datetime(sel["asof"]).to_numpy()
        out.loc[d_has, spec.clim_n_col] = sel["clim_n"].to_numpy(float)
        for base, c in zip(SAT_VALUE_BASE, spec.value_cols):
            out.loc[d_has, c] = np.where(fresh, sel[base].to_numpy(float), np.nan)
        frames.append(out)
    res = pd.concat(frames)
    res.index.name = "date"
    res[spec.asof_col] = pd.to_datetime(res[spec.asof_col])
    return res


def _modis_state(comps: pd.DataFrame, seasons) -> pd.DataFrame:
    """Daily MODIS features for one state (comps = prepared rows of that state)."""
    return _sat_state(comps, seasons, MODIS_SPEC)


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
    """US rows = renormalised fixed-weight mean of state values (per date and column).

    For every satellite source present in ``states_panel`` (its ``anom_col`` is a column)
    adds ``<name>_coverage`` (weight share of states with that anomaly) and then
    ``us_n_states_<name>``; ``us_n_states_weather`` when weather is present."""
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
    specs = [s for s in SAT_SPECS if s.anom_col in p]
    for s in specs:
        res[s.coverage_col] = (w.where(p[s.anom_col].notna(), 0.0)
                               .groupby(p["date"]).sum() / sum(US_WEIGHTS.values()))
    # number of states entering the aggregate (renormalised weights: fewer than 5 means the
    # aggregate is a sub-national mean); satellite counted on its ndvi_anom, weather on
    # gdd_cum_anom
    for s in specs:
        res[s.n_states_col] = p[s.anom_col].notna().groupby(p["date"]).sum().astype(int)
    if "gdd_cum_anom" in p:
        res["us_n_states_weather"] = (p["gdd_cum_anom"].notna().groupby(p["date"]).sum()
                                      .astype(int))
    return res.reset_index()


def _sat_block(prepared: pd.DataFrame, seasons, spec: SatSpec) -> pd.DataFrame:
    """Long (geo, date) frame of one satellite source for the states and the US, columns
    ``["geo", "date", *spec.block_cols]`` (state coverage = 1/0, US-only n_states NaN on
    state rows), built exactly like the MODIS part of the panel."""
    parts = []
    for st in STATES:
        f = _sat_state(prepared[prepared["state"] == st], seasons, spec)
        f.insert(0, "geo", st)
        parts.append(f.reset_index())
    states_panel = pd.concat(parts, ignore_index=True)
    us = _us_aggregate(states_panel, spec.cols, [spec.asof_col])
    us.insert(0, "geo", "US")
    blk = pd.concat([states_panel, us], ignore_index=True)
    st_rows = blk["geo"] != "US"
    blk.loc[st_rows, spec.coverage_col] = np.where(
        blk.loc[st_rows, spec.anom_col].notna(), 1.0, 0.0)
    return blk[["geo", "date", *spec.block_cols]]


def _append_sat(panel: pd.DataFrame, sat: pd.DataFrame, seasons, spec: SatSpec
                ) -> pd.DataFrame:
    """``panel`` with the block of ``spec`` appended as its LAST columns, aligned on
    (geo, date). Existing columns are not touched."""
    blk = _sat_block(prepare_sat(sat, spec), seasons, spec).set_index(["geo", "date"])
    key = pd.MultiIndex.from_arrays([panel["geo"], panel["date"]])
    blk = blk.reindex(key)
    add = pd.DataFrame({c: blk[c].to_numpy() for c in spec.block_cols}, index=panel.index)
    return pd.concat([panel, add], axis=1)


def build_features(modis: pd.DataFrame, weather: pd.DataFrame, usda: dict | None = None,
                   seasons=None, hls: pd.DataFrame | None = None,
                   hls_spec: SatSpec | None = None) -> pd.DataFrame:
    """Daily as-of panel (long; see module doc) from PRE-LOADED input frames.

    ``modis``: modis_state rows (sample and pre-sample seasons concatenated; the stored
    available_date is ignored and replaced by end_date + MODIS_LAG_DAYS). ``weather``:
    weather_daily rows (sample + pre-sample). ``usda``: optional dict with
    ``condition`` / ``progress`` frames as produced by ``usda.parse_*``. ``seasons``:
    output seasons (default config.YEARS). ``hls``: optional hls_state rows (sample +
    pre-sample flagged ``presample=True``; available = end_date + HLS_LAG_DAYS); its
    ``HLS_SPEC.block_cols`` are appended after every other column, so ``hls=None`` gives
    exactly the MODIS-only panel. Every input row is used only on dates >= its
    availability date, so passing frames truncated at date d yields identical values
    for all rows dated <= d (the truncation test relies on this).
    ``hls_spec``: override of ``HLS_SPEC`` for the HLS block (e.g.
    ``dataclasses.replace(HLS_SPEC, lag_days=14)`` for a lag-sensitivity run); None (the
    default) uses ``HLS_SPEC`` unchanged. The block's column names come from the spec, so
    an override must keep ``name`` / ``prefix``.
    """
    seasons = list(YEARS if seasons is None else seasons)
    mp = prepare_sat(modis, MODIS_SPEC)
    wp = prepare_weather(weather)
    up = prepare_usda(usda)

    parts = []
    for st in STATES:
        f_m = _sat_state(mp[mp["state"] == st], seasons, MODIS_SPEC)
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
    if hls is not None:
        panel = _append_sat(panel, hls, seasons, HLS_SPEC if hls_spec is None else hls_spec)
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


def load_hls(presample: bool = True) -> pd.DataFrame | None:
    """hls_state rows (``presample=False``) plus, if ``presample`` and the file exists,
    the 2013-2015 pre-sample rows (``presample=True``: climatology only). Pre-sample
    seasons that are also in the sample are dropped (the sample wins). None when
    hls_state.parquet does not exist."""
    if not HLS_STATE_PATH.exists():
        return None
    h = pd.read_parquet(HLS_STATE_PATH).assign(presample=False)
    if presample:
        if HLS_PRESAMPLE_PATH.exists():
            hp = pd.read_parquet(HLS_PRESAMPLE_PATH)
            dup = hp["year"].isin(set(h["year"]))
            if dup.any():
                log.warning("HLS pre-sample seasons %s also in the sample; using the sample",
                            sorted(set(hp.loc[dup, "year"])))
            h = pd.concat([hp[~dup].assign(presample=True), h], ignore_index=True)
        else:
            log.warning("no HLS pre-sample; HLS climatology needs %d sample seasons",
                        MIN_CLIM_YEARS)
    return h


def load_inputs(presample: bool = True, hls: bool = True) -> dict:
    """Pre-loaded inputs for ``build_features``: modis (with ``presample`` flag column),
    weather (sample + pre-sample), usda (dict or None) and hls (``load_hls``; None when
    absent or ``hls`` is False)."""
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
    return {"modis": m, "weather": w, "usda": load_usda_tables(),
            "hls": load_hls(presample) if hls else None}


def build_trading(panel: pd.DataFrame, zcz: pd.DataFrame) -> pd.DataFrame:
    """US features (unsuffixed) plus every state's features (suffix ``_<state>``, lower
    case) joined onto front-December ZCZ trading dates that fall in the season window of
    their own campaign (Apr 1 - Oct 31 of year == campaign). Features on row d are as of
    d (see module doc). The columns of an appended satellite block (HLS) come last, US
    then states, so the table without HLS is exactly the MODIS-only table."""
    z = zcz[["date", "contract", "campaign", "close", "ret", "days_to_expiry"]].copy()
    z["date"] = _dt(z["date"])
    z = z[z["campaign"] == z["date"].dt.year]
    feat_cols = [c for c in panel.columns if c not in ("geo", "date", "season", "doy")]
    appended = [s.block_cols for s in SAT_SPECS if s is not MODIS_SPEC]
    in_blocks = {c for b in appended for c in b}
    groups = [[c for c in feat_cols if c not in in_blocks],
              *[[c for c in feat_cols if c in b] for b in appended]]
    out = z
    for i, cols in enumerate(groups):
        if i and not cols:
            continue
        us = panel[panel["geo"] == "US"][["date", *cols]]
        out = out.merge(us, on="date", how="inner" if i == 0 else "left")
        st_cols = [c for c in cols if c not in US_ONLY_COLS]
        for st in STATES:
            s = panel[panel["geo"] == st][["date", *st_cols]]
            s = s.rename(columns={c: f"{c}_{st.lower()}" for c in st_cols})
            out = out.merge(s, on="date", how="left")
    return out.sort_values("date").reset_index(drop=True)


def build_and_save(presample: bool = True, hls: bool = True
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    inp = load_inputs(presample, hls=hls)
    panel = build_features(inp["modis"], inp["weather"], inp["usda"], hls=inp["hls"])
    zcz = pd.read_parquet(DATA_PROCESSED / "zcz_front_dec.parquet")
    trading = build_trading(panel, zcz)
    FEATURES_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(FEATURES_DAILY_PATH, index=False)
    trading.to_parquet(FEATURES_TRADING_PATH, index=False)
    return panel, trading

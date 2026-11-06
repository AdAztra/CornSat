"""USDA NASS Quick Stats: crop condition, progress, yield; with release-date alignment.

Every output table carries ``available_date``: the first date on which the
number could have been known at the market open, so it can be used without
look-ahead in a backtest.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

from cornsat.config import DATA_INTERIM, DATA_PROCESSED, ROOT, STATES, YEARS

API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
CACHE_DIR = DATA_INTERIM / "usda"
KEY_HELP = (
    "NASS_API_KEY is not set. Get a free key at https://quickstats.nass.usda.gov/api "
    f"and put the line NASS_API_KEY=<your key> in {ROOT / '.env'}."
)

_HOLIDAYS = USFederalHolidayCalendar().holidays(start="2000-01-01", end="2040-12-31")
_BDAY = CustomBusinessDay(holidays=_HOLIDAYS)

COND_UNITS = {
    "PCT VERY POOR": "pct_vp",
    "PCT POOR": "pct_p",
    "PCT FAIR": "pct_f",
    "PCT GOOD": "pct_g",
    "PCT EXCELLENT": "pct_e",
}
PROGRESS_STAGES = ["PLANTED", "EMERGED", "SILKING", "DOUGH", "DENTED", "MATURE", "HARVESTED"]
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}


class NassKeyMissing(RuntimeError):
    pass


def get_api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.environ.get("NASS_API_KEY", "").strip()
    if not key:
        raise NassKeyMissing(KEY_HELP)
    return key


def parse_value(v) -> float:
    """'1,234' -> 1234.0; '(D)', '(NA)', '(Z)', '' , None -> NaN."""
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s or s.startswith("("):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _cache_path(params: dict):
    blob = json.dumps(params, sort_keys=True, default=str)
    return CACHE_DIR / f"{hashlib.md5(blob.encode()).hexdigest()}.parquet"


def query(retries: int = 4, use_cache: bool = True, **params) -> pd.DataFrame:
    """Query Quick Stats; returns raw records as a DataFrame (all columns str).

    Cached as parquet under data/interim/usda/ (key excluded from the cache key).
    A 400 'exceeds limit' response means >50,000 rows: narrow the query.
    No matching data (HTTP 400 'no data') returns an empty frame.
    """
    params = {k: v for k, v in params.items() if v is not None}
    path = _cache_path(params)
    if use_cache and path.exists():
        return pd.read_parquet(path)
    full = {**params, "format": "JSON", "key": get_api_key()}
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=full, timeout=60)
        except requests.RequestException as e:
            last = e
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            df = pd.DataFrame(r.json().get("data", []))
            break
        msg = r.text[:300]
        if r.status_code == 400:
            if "exceeds" in msg.lower():
                raise RuntimeError(f"NASS query returns >50,000 rows; narrow it: {params}")
            if "no data" in msg.lower():
                df = pd.DataFrame()
                break
            raise RuntimeError(f"NASS API rejected the request (400): {msg} params={params}")
        if r.status_code in (401, 403):
            raise RuntimeError(f"NASS API key rejected ({r.status_code}). {KEY_HELP}")
        last = RuntimeError(f"HTTP {r.status_code}: {msg}")
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"NASS query failed after {retries} tries: {last}; params={params}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.astype(str).to_parquet(path)
    return df


# ---------------------------------------------------------------- release dates

def crop_progress_available_date(week_ending) -> pd.Timestamp:
    """Weekly Crop Progress: Sunday week_ending, released Monday 4pm ET -> usable Tuesday.

    If Monday is a federal holiday, released Tuesday 4pm -> usable Wednesday.
    """
    we = pd.Timestamp(week_ending).normalize()
    monday = we + pd.Timedelta(days=1)
    return we + pd.Timedelta(days=3 if monday in _HOLIDAYS else 2)


def crop_production_release_date(year: int, month: int) -> pd.Timestamp:
    """Approximate *usable* date of the monthly Crop Production report.

    APPROXIMATION: NASS publishes ~10th-12th at noon ET. We take the first
    business day (non-weekend, non-federal-holiday) on or after the 10th as the
    release day, and make data available the NEXT business day (conservative,
    since the noon release comes mid-session). Exact historical dates can
    replace this later.
    """
    d = pd.Timestamp(year=year, month=month, day=10)
    release = d if d.weekday() < 5 and d not in _HOLIDAYS else d + _BDAY
    return release + _BDAY


def final_yield_available_date(year: int) -> pd.Timestamp:
    """Final yield for crop year Y is in January Annual Crop Production of Y+1 (TARGET only)."""
    return crop_production_release_date(year + 1, 1)


def next_business_day_after(ts) -> pd.Timestamp:
    """First weekday, non-federal-holiday date strictly after ts's calendar date."""
    return pd.Timestamp(ts).normalize() + _BDAY


def today() -> pd.Timestamp:
    """Injectable 'today' (tests monkeypatch ``usda.today``)."""
    return pd.Timestamp.today().normalize()


# Crop Production report release days for 2009-2011, where NASS Quick Stats ``load_time`` is a
# bulk load (2012-01-01) and cannot date the release. (year, month) -> release day.
# TODO verify: entered FROM MEMORY, not checked against the NASS Crop Production archive.
RELEASE_CALENDAR: dict[tuple[int, int], pd.Timestamp] = {
    (y, m): pd.Timestamp(y, m, d)
    for y, days in {
        2009: {8: 12, 9: 11, 10: 9, 11: 10},
        2010: {8: 12, 9: 10, 10: 8, 11: 9},
        2011: {8: 11, 9: 12, 10: 12, 11: 9},
    }.items()
    for m, d in days.items()
}


def forecast_available_date(year: int, month: int, load_times=None) -> pd.Timestamp:
    """Forecast usable date = max of (rule date, next business day after the
    RELEASE_CALENDAR day when listed, next business day after the (max) load_time date when
    it parses and lies in the forecast's year/month)."""
    rule = crop_production_release_date(year, month)
    cal = RELEASE_CALENDAR.get((year, month))
    if cal is not None:
        rule = max(rule, next_business_day_after(cal))
    lt = pd.to_datetime(pd.Series([] if load_times is None else list(load_times)),
                        errors="coerce").dropna()
    lt = lt[(lt.dt.year == year) & (lt.dt.month == month)]
    if lt.empty:
        return rule
    return max(rule, next_business_day_after(lt.max()))


# ---------------------------------------------------------------- parsers

def _geo(df: pd.DataFrame):
    return np.where(df["agg_level_desc"] == "NATIONAL", "US", df["state_alpha"])


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["value"] = df["Value"].map(parse_value)
    df["geo"] = _geo(df)
    return df


def condition_index(vp, p, f, g, e):
    return (1 * vp + 2 * p + 3 * f + 4 * g + 5 * e) / 100


def parse_condition(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["geo", "week_ending", "pct_vp", "pct_p", "pct_f", "pct_g", "pct_e", "ge", "ci", "available_date"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = _prep(df)
    df = df[df["unit_desc"].isin(COND_UNITS) & (df["statisticcat_desc"] == "CONDITION")].copy()
    df["col"] = df["unit_desc"].map(COND_UNITS)
    wide = df.pivot_table(index=["geo", "week_ending"], columns="col", values="value", aggfunc="first")
    wide = wide.reindex(columns=list(COND_UNITS.values())).dropna().reset_index()
    wide.columns.name = None
    wide["ge"] = wide["pct_g"] + wide["pct_e"]
    wide["ci"] = condition_index(wide.pct_vp, wide.pct_p, wide.pct_f, wide.pct_g, wide.pct_e)
    wide["week_ending"] = pd.to_datetime(wide["week_ending"])
    wide["available_date"] = wide["week_ending"].map(crop_progress_available_date)
    return wide[cols].sort_values(["geo", "week_ending"]).reset_index(drop=True)


def parse_progress(df: pd.DataFrame) -> pd.DataFrame:
    stages = [f"pct_{s.lower()}" for s in PROGRESS_STAGES]
    cols = ["geo", "week_ending", *stages, "available_date"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = _prep(df)
    df = df[df["statisticcat_desc"] == "PROGRESS"].copy()
    df["col"] = "pct_" + df["unit_desc"].str.replace("PCT ", "", regex=False).str.lower()
    df = df[df["col"].isin(stages)]
    wide = df.pivot_table(index=["geo", "week_ending"], columns="col", values="value",
                          aggfunc="first", dropna=False)
    wide = wide.reindex(columns=stages).reset_index()
    wide.columns.name = None
    wide["week_ending"] = pd.to_datetime(wide["week_ending"])
    wide["available_date"] = wide["week_ending"].map(crop_progress_available_date)
    return wide[cols].sort_values(["geo", "week_ending"]).reset_index(drop=True)


_STAT_COL = {
    ("YIELD", "BU / ACRE"): "yield_bu_ac",
    ("PRODUCTION", "BU"): "production_bu",
    ("AREA HARVESTED", "ACRES"): "area_harvested_ac",
}


def parse_yield(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (geo, year, period); period is 'FINAL' or the forecast month 'AUG'..'NOV'."""
    cols = ["geo", "year", "period", "yield_bu_ac", "production_bu", "area_harvested_ac", "available_date"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = _prep(df)
    df = df[(df["source_desc"] == "SURVEY") & (df["commodity_desc"] == "CORN")]
    if "util_practice_desc" in df:
        df = df[df["util_practice_desc"] == "GRAIN"]
    df = df.copy()
    df["col"] = [_STAT_COL.get((s, u)) for s, u in zip(df["statisticcat_desc"], df["unit_desc"])]
    ref = df["reference_period_desc"]
    keep = df["col"].notna() & ((ref == "YEAR") | ref.str.match(r"^YEAR - [A-Z]{3} FORECAST$"))
    df = df[keep].copy()
    df["period"] = np.where(df["reference_period_desc"] == "YEAR", "FINAL",
                            df["reference_period_desc"].str.slice(7, 10))
    df["year"] = df["year"].astype(int)
    keys = ["geo", "year", "period"]
    wide = df.pivot_table(index=keys, columns="col", values="value",
                          aggfunc="first", dropna=False)
    wide = wide.reindex(columns=["yield_bu_ac", "production_bu", "area_harvested_ac"]).reset_index()
    wide.columns.name = None
    if "load_time" in df:
        df["_lt"] = pd.to_datetime(df["load_time"], errors="coerce")
        lts = df.groupby(keys)["_lt"].apply(list).to_dict()
    else:
        lts = {}

    def avail(r):
        if r["period"] == "FINAL":
            return final_yield_available_date(r["year"])
        return forecast_available_date(r["year"], MONTHS[r["period"]],
                                       lts.get((r["geo"], r["year"], r["period"])))

    # "YEAR" is not final during the season: drop FINAL rows not yet released.
    t = today()
    is_final = wide["period"] == "FINAL"
    unreleased = is_final & ((wide["year"] >= t.year) | wide["year"].map(
        lambda y: final_yield_available_date(y) > t))
    wide = wide[~unreleased].copy()
    wide["available_date"] = wide.apply(avail, axis=1)
    return wide[cols].sort_values(["geo", "year", "available_date"]).reset_index(drop=True)


# ---------------------------------------------------------------- fetchers

def _base(agg: str, years) -> dict:
    p = {"source_desc": "SURVEY", "commodity_desc": "CORN", "agg_level_desc": agg,
         "year__GE": min(years), "year__LE": max(years)}
    if agg == "STATE":
        p["state_alpha"] = list(STATES)
    return p


def _fetch(agg_levels, years, **extra) -> pd.DataFrame:
    frames = [query(**_base(agg, years), **extra) for agg in agg_levels]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _concat(frames):
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def crop_condition(years=YEARS) -> pd.DataFrame:
    # Per call: source SURVEY, commodity CORN, statisticcat CONDITION, unit_desc = PCT <category>
    # (5 calls x [STATE with state_alpha in 5 states, NATIONAL]), year__GE/LE over the range.
    return parse_condition(_concat(
        [_fetch(("STATE", "NATIONAL"), years, statisticcat_desc="CONDITION", unit_desc=u)
         for u in COND_UNITS]))


def crop_progress(years=YEARS) -> pd.DataFrame:
    # Per call: statisticcat PROGRESS, unit_desc = PCT <stage> (7 stages x [STATE, NATIONAL]).
    return parse_progress(_concat(
        [_fetch(("STATE", "NATIONAL"), years, statisticcat_desc="PROGRESS", unit_desc=f"PCT {s}")
         for s in PROGRESS_STAGES]))


def annual_yield(years=YEARS) -> pd.DataFrame:
    # Per call: statisticcat YIELD/PRODUCTION/AREA HARVESTED with unit_desc as in _STAT_COL,
    # util_practice_desc GRAIN. No reference_period filter: "YEAR" (final) and
    # "YEAR - XXX FORECAST" rows are selected client-side.
    return parse_yield(_concat(
        [_fetch(("STATE", "NATIONAL"), years, statisticcat_desc=s, unit_desc=u,
                util_practice_desc="GRAIN") for (s, u) in _STAT_COL]))


def build_usda() -> dict:
    get_api_key()
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out = {"usda_condition": crop_condition(), "usda_progress": crop_progress(),
           "usda_yield": annual_yield()}
    for name, df in out.items():
        df.to_parquet(DATA_PROCESSED / f"{name}.parquet", index=False)
    return out

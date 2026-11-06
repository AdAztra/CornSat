"""Daily weather features from NASA POWER, averaged over corn-belt points."""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import requests

from cornsat.config import DATA_INTERIM, DATA_PROCESSED, SEASON_END, SEASON_START, STATES

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
START = "20080101"
WEATHER_DIR = DATA_INTERIM / "weather"

# Approximate county-seat coordinates (name, lat, lon) of top corn counties,
# spread across each state. Equal weights. Coordinates are approximate.
CORN_POINTS: dict[str, list[tuple[str, float, float]]] = {
    "IA": [
        ("Kossuth/Algona", 43.07, -94.23),
        ("Sioux/Orange City", 43.00, -96.06),
        ("Plymouth/Le Mars", 42.79, -96.17),
        ("Hamilton/Webster City", 42.47, -93.82),
        ("Story/Nevada", 42.02, -93.45),
        ("Benton/Vinton", 42.17, -92.02),
        ("Washington/Washington", 41.30, -91.69),
    ],
    "IL": [
        ("McLean/Bloomington", 40.48, -88.99),
        ("Champaign/Urbana", 40.11, -88.21),
        ("Livingston/Pontiac", 40.88, -88.63),
        ("LaSalle/Ottawa", 41.35, -88.84),
        ("Iroquois/Watseka", 40.78, -87.74),
        ("Henry/Cambridge", 41.30, -90.19),
        ("Adams/Quincy", 39.94, -91.41),
    ],
    "NE": [
        ("York/York", 40.87, -97.59),
        ("Platte/Columbus", 41.43, -97.37),
        ("Buffalo/Kearney", 40.70, -99.08),
        ("Custer/Broken Bow", 41.40, -99.64),
        ("Hall/Grand Island", 40.92, -98.34),
        ("Cuming/West Point", 41.84, -96.71),
        ("Antelope/Neligh", 42.13, -98.03),
    ],
    "MN": [
        ("Blue Earth/Mankato", 44.16, -93.99),
        ("Renville/Olivia", 44.08, -94.99),
        ("Martin/Fairmont", 43.65, -94.46),
        ("Stearns/St Cloud", 45.56, -94.16),
        ("Nicollet/St Peter", 44.32, -93.96),
        ("Freeborn/Albert Lea", 43.65, -93.37),
        ("Lyon/Marshall", 44.45, -95.79),
    ],
    "IN": [
        ("Tippecanoe/Lafayette", 40.42, -86.88),
        ("Jasper/Rensselaer", 40.94, -87.15),
        ("Carroll/Delphi", 40.59, -86.68),
        ("Rush/Rushville", 39.61, -85.45),
        ("Daviess/Washington", 38.66, -87.17),
        ("Henry/New Castle", 39.93, -85.37),
        ("Allen/Fort Wayne", 41.08, -85.14),
    ],
}

# Approximate production shares (fixed) for the national aggregate.
US_WEIGHTS = {"IA": 0.30, "IL": 0.26, "NE": 0.20, "MN": 0.14, "IN": 0.10}

GDD_BASE = 10.0
GDD_CAP = 30.0
EDD_THRESHOLD = 30.0


def fetch_point(lat: float, lon: float, force: bool = False, retries: int = 5) -> pd.DataFrame:
    """Daily T2M_MAX/T2M_MIN/PRECTOTCORR for a point (cached as parquet, refreshed daily)."""
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    path = WEATHER_DIR / f"point_{lat:.2f}_{lon:.2f}.parquet"
    if path.exists() and not force:
        if time.time() - path.stat().st_mtime < 24 * 3600:
            return pd.read_parquet(path)
    params = {
        "parameters": "T2M_MAX,T2M_MIN,PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": START,
        "end": f"{date.today():%Y%m%d}",
        "format": "JSON",
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.get(POWER_URL, params=params, timeout=120)
            r.raise_for_status()
            par = r.json()["properties"]["parameter"]
            break
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(2 ** (attempt + 1))
    else:
        raise RuntimeError(f"NASA POWER fetch failed for {lat},{lon}: {last_err}")
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(list(par["T2M_MAX"].keys()), format="%Y%m%d"),
            "tmax_c": list(par["T2M_MAX"].values()),
            "tmin_c": list(par["T2M_MIN"].values()),
            "precip_mm": list(par["PRECTOTCORR"].values()),
        }
    )
    df = df.replace(-999, np.nan)  # POWER missing-value sentinel
    df.to_parquet(path, index=False)
    time.sleep(1.0)
    return df


def state_daily(state: str) -> pd.DataFrame:
    """Equal-weighted (NaN-skipping) mean over the state's points; trailing all-NaN days dropped."""
    frames = [fetch_point(lat, lon) for _, lat, lon in CORN_POINTS[state]]
    avg = pd.concat(frames).groupby("date").mean()
    avg = avg.sort_index().dropna(how="all").reset_index()
    return avg[["date", "tmax_c", "tmin_c", "precip_mm"]]


def compute_gdd(tmax, tmin):
    """Corn GDD (C): each of Tmax, Tmin clipped to [10, 30]; mean minus base 10."""
    tmax = pd.Series(tmax, dtype=float).clip(GDD_BASE, GDD_CAP)
    tmin = pd.Series(tmin, dtype=float).clip(GDD_BASE, GDD_CAP)
    return (tmax + tmin) / 2 - GDD_BASE


def compute_edd30(tmax):
    """Simple extreme degree days: max(Tmax - 30, 0) (daily-max approximation)."""
    return (pd.Series(tmax, dtype=float) - EDD_THRESHOLD).clip(lower=0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add gdd, edd30, season cumulatives, precip windows and available_date.

    Expects columns date, tmax_c, tmin_c, precip_mm for a single series.
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["gdd"] = compute_gdd(df["tmax_c"], df["tmin_c"]).values
    df["edd30"] = compute_edd30(df["tmax_c"]).values
    md = df["date"].dt.month * 100 + df["date"].dt.day
    lo = SEASON_START[0] * 100 + SEASON_START[1]
    hi = SEASON_END[0] * 100 + SEASON_END[1]
    in_season = (md >= lo) & (md <= hi)
    yr = df["date"].dt.year
    for col in ("gdd", "edd30"):
        s = df[col].where(in_season)
        df[f"{col}_cum"] = s.groupby(yr).cumsum().where(in_season)
    df["precip_30d"] = df["precip_mm"].rolling(30, min_periods=30).sum()
    p = df["precip_mm"].where(in_season)
    df["precip_cum"] = p.groupby(yr).cumsum().where(in_season)
    df["available_date"] = df["date"] + pd.Timedelta(days=1)
    return df


def build_weather(states: dict = STATES, save: bool = True) -> pd.DataFrame:
    """Long table (state, date, ...) incl. national aggregate state='US'."""
    parts, daily = [], {}
    for st in states:
        daily[st] = state_daily(st)
        d = add_features(daily[st])
        d.insert(0, "state", st)
        parts.append(d)
    w = {s: US_WEIGHTS[s] for s in states if s in US_WEIGHTS}
    tot = sum(w.values())
    cols = ["tmax_c", "tmin_c", "precip_mm"]
    us = sum(daily[s].set_index("date")[cols] * (w[s] / tot) for s in w)
    us = us.dropna().reset_index()  # only dates where every weighted state has data
    d = add_features(us)
    d.insert(0, "state", "US")
    parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out.to_parquet(DATA_PROCESSED / "weather_daily.parquet", index=False)
    return out

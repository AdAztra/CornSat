"""Build the Phase 5 daily as-of feature panel and the ZCZ trading-date table.

Usage:
    uv run python scripts/build_features.py                 # features only
    uv run python scripts/build_features.py --presample     # first build the 2003-2008
                                                            # climatology history (MODIS
                                                            # ~4.5 min/season, resumable;
                                                            # NASA POWER weather 2003-2007)
    uv run python scripts/build_features.py --no-presample  # ignore pre-sample files

Writes data/processed/features_daily.parquet (long: one row per geo x calendar day,
Apr 1 - Oct 31, 2009-2026) and data/processed/features_trading.parquet (US features
plus per-state columns suffixed _<state> on ZCZ front-December trading dates). USDA
features are added automatically when data/processed/usda_*.parquet exist.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from cornsat import features as F


def build_presample() -> None:
    from cornsat import modis, weather

    w = weather.build_weather_presample()
    print(f"weather pre-sample: {len(w)} rows {w['date'].min().date()}..{w['date'].max().date()}")
    m = modis.build_presample()
    print(f"MODIS pre-sample: {len(m)} rows, seasons {sorted(m['year'].unique())}")


def sanity(panel: pd.DataFrame, trading: pd.DataFrame) -> None:
    us = panel[panel["geo"] == "US"].copy()
    late = us[(us["date"].dt.month == 8) | ((us["date"].dt.month == 7) & (us["date"].dt.day >= 20))]
    tbl = late.groupby("season")[["ndvi_anom", "evi_anom", "ndvi_z", "edd30_jul"]].mean()
    print("\nUS mean anomalies, Jul 20 - Aug 31:")
    print(tbl.round(4).to_string())
    a1 = us[(us["date"].dt.month == 8) & (us["date"].dt.day == 1)].set_index("season")
    ok = a1[["ndvi_anom", "edd30_jul"]].dropna()
    print(f"\ncorr(US Aug-1 ndvi_anom, US Aug-1 edd30_jul) over {len(ok)} seasons: "
          f"{ok.corr().iloc[0, 1]:.3f}")
    s12 = us[us["season"] == 2012]
    first = s12[s12["ndvi_z"] < -1]
    print("2012 first date US ndvi_z < -1:",
          first["date"].min().date() if len(first) else None)
    print("rows:", len(panel), "| trading rows:", len(trading))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--presample", action="store_true", help="build the pre-sample history first")
    p.add_argument("--no-presample", action="store_true", help="do not use pre-sample files")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.presample:
        build_presample()
    panel, trading = F.build_and_save(presample=not args.no_presample)
    for c in ("modis_asof", "weather_asof", "usda_asof"):
        if c in panel:
            assert (panel[c].isna() | (panel[c] <= panel["date"])).all(), c
    print(f"wrote {F.FEATURES_DAILY_PATH} {panel.shape} and {F.FEATURES_TRADING_PATH} "
          f"{trading.shape}; USDA features: {'usda_asof' in panel}")
    with np.errstate(all="ignore"):
        sanity(panel, trading)


if __name__ == "__main__":
    main()

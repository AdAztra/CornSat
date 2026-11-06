"""Build and cache corn-fraction rasters (MODIS grid) from USDA CDL.

Usage:
    uv run python scripts/build_cornmask.py --states IA IL --years 2011 2012 2013
Defaults: the 5 core states, CDL years 2008-2025.
"""

from __future__ import annotations

import argparse
import time

from cornsat import cornmask
from cornsat.config import STATES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", nargs="+", default=list(STATES), help="state abbreviations")
    p.add_argument("--years", nargs="+", type=int, default=cornmask.cdl_years(), help="CDL years")
    p.add_argument("--refresh", action="store_true", help="rebuild even if cached")
    args = p.parse_args()

    t_all = time.time()
    print(f"{'state':5} {'year':4} {'corn Macres (MODIS)':>20} {'corn Macres (30m)':>18} "
          f"{'src':>22} {'sec':>6} {'dl MB':>8}", flush=True)
    for st in args.states:
        for yr in args.years:
            t0, b0 = time.time(), cornmask.DOWNLOAD_STATS["bytes"]
            da = cornmask.corn_fraction(st, yr, refresh=args.refresh)
            acres = cornmask.corn_acres(da) / 1e6
            a30 = da.attrs.get("corn_acres_30m", float("nan")) / 1e6
            dl = (cornmask.DOWNLOAD_STATS["bytes"] - b0) / 1e6
            print(f"{st:5} {yr:4d} {acres:20.3f} {a30:18.3f} {cornmask.cdl_source(yr):>22} "
                  f"{time.time() - t0:6.1f} {dl:8.1f}", flush=True)
    print(f"total downloaded: {cornmask.DOWNLOAD_STATS['bytes'] / 1e6:.1f} MB; "
          f"elapsed {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()

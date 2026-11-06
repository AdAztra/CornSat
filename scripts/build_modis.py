"""Build corn-weighted MODIS NDVI/EVI per state and 16-day composite (Terra + Aqua).

Usage:
    uv run python scripts/build_modis.py --years 2011 2012 2013 --states IA IL
Defaults: the 5 core states, seasons 2009-2026. Years already cached in
data/interim/modis/{year}.parquet are skipped (resumable); the cached years requested
are then concatenated into data/processed/modis_state.parquet.
"""

from __future__ import annotations

import argparse
import logging
import time

from cornsat import modis
from cornsat.config import STATES, YEARS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--years", nargs="+", type=int, default=list(YEARS), help="season years")
    p.add_argument("--states", nargs="+", default=list(STATES), help="state abbreviations")
    p.add_argument("--workers", type=int, default=8, help="dask threads per composite load")
    p.add_argument("--parallel", type=int, default=3, help="composites loaded concurrently")
    p.add_argument("--no-combine", action="store_true",
                   help="only build per-year caches; do not write the processed parquet")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    unknown = [s for s in args.states if s not in STATES]
    if unknown:
        p.error(f"unknown states {unknown}; choose from {list(STATES)}")
    states = {s: STATES[s] for s in args.states}

    t0 = time.time()
    if args.no_combine:
        for y in args.years:
            modis.build_year(y, states, workers=args.workers, parallel=args.parallel)
    else:
        df = modis.build_all(args.years, states, workers=args.workers, parallel=args.parallel)
        print(f"wrote {modis.OUT_PATH}: {len(df)} rows, "
              f"{df['year'].min()}-{df['year'].max()}, mean valid_frac {df['valid_frac'].mean():.3f}")
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

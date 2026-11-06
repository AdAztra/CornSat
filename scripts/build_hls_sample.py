"""Build the HLS block sample (blocks.parquet) and per-season predicted-corn masks.

Usage:
    uv run python scripts/build_hls_sample.py --blocks-only
    uv run python scripts/build_hls_sample.py --states IA --seasons 2023 2024
Defaults: the 5 core states, seasons 2013-2026. The block table is always built jointly
for all core states (``--states`` only restricts the masks). Purity (an ex-post
diagnostic) is printed for seasons whose CDL exists.
"""

from __future__ import annotations

import argparse
import time

from cornsat import cornmask, hlssample
from cornsat.config import STATES


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--states", nargs="+", default=list(STATES), help="state abbreviations")
    p.add_argument("--seasons", nargs="+", type=int, default=list(range(2013, 2027)))
    p.add_argument("--blocks-only", action="store_true", help="only build tiles/blocks tables")
    p.add_argument("--refresh", action="store_true", help="rebuild masks even if cached")
    p.add_argument("--refresh-blocks", action="store_true", help="rebuild the block table")
    args = p.parse_args()

    t0 = time.time()
    blocks = hlssample.build_blocks(refresh=args.refresh_blocks)
    km2 = (blocks["height"] * blocks["width"]).sum() * hlssample.HLS_RES**2 / 1e6
    print(f"blocks: {len(blocks)} ({km2:,.0f} km2) in {time.time() - t0:.1f}s")
    for st, g in blocks.groupby("state", sort=False):
        print(f"  {st}: {g['tile'].nunique()} tiles, {len(g)} blocks, "
              f"mean corn frac 2008-12 {g['corn_frac_0812'].mean():.3f}")
    if args.blocks_only:
        return

    print(f"{'state':5} {'season':6} {'cdl_years':>18} {'corn px':>10} {'purity':>7} {'sec':>6} {'dl MB':>7}")
    for st in args.states:
        for season in args.seasons:
            t1, b1 = time.time(), cornmask.DOWNLOAD_STATS["bytes"]
            hlssample.build_masks(season, [st], refresh=args.refresh)
            meta = hlssample.mask_meta(st, season)
            ids = hlssample.load_blocks(st)["block_id"]
            n = sum(int(hlssample.season_mask(b, season).sum()) for b in ids)
            pur = (
                f"{hlssample.mask_purity(st, season):.3f}"
                if season <= cornmask.CDL_LAST_YEAR
                else "   n/a"
            )
            dl = (cornmask.DOWNLOAD_STATS["bytes"] - b1) / 1e6
            print(f"{st:5} {season:6d} {str(meta['cdl_years']):>18} {n:10d} {pur:>7} "
                  f"{time.time() - t1:6.1f} {dl:7.1f}", flush=True)
    print(f"elapsed {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

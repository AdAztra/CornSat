"""Ingest NASA HLS v2.0 over the fixed block sample and build causal state composites.

Usage:
    uv run python scripts/build_hls.py --pilot                 # IA, season 2024 only
    uv run python scripts/build_hls.py --years 2016 2017 --states IA IL
    uv run python scripts/build_hls.py                         # 5 states x 2013-2026
    uv run python scripts/build_hls.py --combine-only          # composites from the cache

Ingestion is resumable (one scene cache per year and tile under data/interim/hls/scenes).
The season masks must exist (scripts/build_hls_sample.py). Outputs:
data/processed/hls_state.parquet (2016+) and hls_state_presample.parquet (2013-2015).
The pilot writes data/interim/hls/pilot_IA_2024.parquet instead and prints diagnostics.
No URL, header or token is ever printed.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from cornsat import hls
from cornsat.config import DATA_PROCESSED, STATES


def print_summary(s: dict) -> None:
    el = s["elapsed_s"]
    print(
        f"[{s['year']} {','.join(s['states'])}] tiles {s['tiles']}, blocks {s['blocks']} | "
        f"granules found {s['granules_found']}, after cloud prefilter {s['granules_prefiltered']}, "
        f"cached {s['granules_cached']}, to do {s['granules_todo']}: read {s['granules_read']}, "
        f"skipped cloudy {s['granules_skipped_cloudy']}, failed {s['granules_failed']} "
        f"{s['errors'] or ''}"
    )
    rq = s["requests"]
    print(
        f"    requests {rq} (resolve {s['resolve_requests']}, open {s['opens']}, window reads "
        f"{s['window_reads']}, retries {s['retries']}) in {el:.0f}s = {rq / max(el, 1e-9):.1f} req/s, "
        f"{s['granules_todo'] / max(el, 1e-9):.2f} granules/s"
    )


def modis_compare(comps: pd.DataFrame, state: str, year: int) -> None:
    path = DATA_PROCESSED / "modis_state.parquet"
    if not path.exists():
        print("no modis_state.parquet; skipping the MODIS comparison")
        return
    m = pd.read_parquet(path)
    m = m[(m["state"] == state) & (m["year"] == year)].copy()
    m["end_date"] = pd.to_datetime(m["end_date"]).astype("datetime64[ns]")
    m = m.groupby("end_date", as_index=False)["ndvi_mean"].mean().sort_values("end_date")
    h = comps[(comps["state"] == state) & (comps["year"] == year)].copy()
    h = h.dropna(subset=["ndvi_mean"]).sort_values("end_date")
    j = pd.merge_asof(h[["end_date", "ndvi_mean", "valid_frac"]], m.rename(
        columns={"ndvi_mean": "modis_ndvi", "end_date": "modis_end"}), left_on="end_date",
        right_on="modis_end", direction="nearest", tolerance=pd.Timedelta(days=2)).dropna()
    print(f"\nHLS vs MODIS {state} {year} (matched on end_date +-2 d):")
    print(j.rename(columns={"ndvi_mean": "hls_ndvi"}).to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    if len(j) > 2:
        print(f"Pearson r = {np.corrcoef(j['ndvi_mean'], j['modis_ndvi'])[0, 1]:.3f} (n={len(j)}), "
              f"mean HLS-MODIS = {(j['ndvi_mean'] - j['modis_ndvi']).mean():+.3f}")


def pilot(workers: int, progress: bool) -> None:
    state, year = "IA", 2024
    s = hls.build_year(year, [state], workers=workers, progress=progress)
    print_summary(s)
    rows = hls.load_scene_rows([year], [state])
    b = hls.season_blocks(year, [state])
    comps = hls.composite(rows, b[["season", "block_id", "state", "tile", "n_mask"]],
                          {(state, year): hls.cdl_years_str(state, year)})
    out = hls.HLS_DIR / f"pilot_{state}_{year}.parquet"
    hls._write_atomic(comps, out)
    ok = comps.dropna(subset=["ndvi_mean"])
    print(f"\ncomposites: {len(comps)} ({len(ok)} with data, "
          f"{int((comps['valid_frac'] >= 0.5).sum())} with valid_frac >= 0.5) -> {out.name}")
    good = comps[comps["valid_frac"] >= 0.5]
    if len(good):
        pk = good.loc[good["ndvi_mean"].idxmax()]
        print(f"NDVI peak (valid_frac >= 0.5): {pk['ndvi_mean']:.3f} on end_date "
              f"{pk['end_date']:%Y-%m-%d} (obs_date {pk['obs_date']:%Y-%m-%d})")
    q = comps["valid_frac"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]).round(3)
    print("valid_frac quantiles:", dict(q))
    print(comps[["end_date", "obs_date", "ndvi_mean", "evi_mean", "ndvi_median", "valid_frac",
                 "n_blocks", "n_l30", "n_s30"]].to_string(index=False,
                                                          float_format=lambda v: f"{v:.3f}"))
    print("\nL30 vs S30 (S30 - L30, block pairs <= 1 day apart):")
    print(hls.sensor_bias(rows).to_string(float_format=lambda v: f"{v:+.4f}"))
    modis_compare(comps, state, year)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--years", nargs="+", type=int, default=list(range(hls.FIRST_YEAR, 2027)))
    p.add_argument("--states", nargs="+", default=list(STATES))
    p.add_argument("--workers", type=int, default=hls.DEFAULT_WORKERS)
    p.add_argument("--pilot", action="store_true", help="IA season 2024 only, with diagnostics")
    p.add_argument("--combine-only", action="store_true", help="no ingestion; composite the cache")
    p.add_argument("--bias-correction", action="store_true",
                   help="subtract the prior-season S30-L30 offset from S30 (default off)")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    progress = not args.no_progress
    t0 = time.time()
    if args.pilot:
        pilot(args.workers, progress)
        print(f"\nelapsed {time.time() - t0:.0f}s")
        return
    summaries = []
    if not args.combine_only:
        for y in args.years:
            s = hls.build_year(y, args.states, workers=args.workers, progress=progress)
            print_summary(s)
            summaries.append(s)
    comps = hls.combine(None, args.states, bias_correction=args.bias_correction)
    written = hls.write_state(comps)
    if summaries:
        tot = {k: sum(s[k] for s in summaries) for k in (
            "granules_found", "granules_todo", "granules_read", "granules_skipped_cloudy",
            "granules_failed", "requests")}
        print(f"\nTOTAL {tot}")
    print(f"composites {len(comps)}; written {written}")
    if len(comps):
        print(hls.sensor_bias(hls.load_scene_rows(None, args.states)).to_string(
            float_format=lambda v: f"{v:+.4f}"))
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

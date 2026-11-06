"""Run the CornSat pipeline end to end.

Steps, in order: prices -> cornmask -> modis -> hls_sample -> hls -> weather -> usda -> features
-> backtest.

Usage:
    uv run python scripts/build_all.py                  # everything; skips expensive steps
                                                        # whose outputs already exist
    uv run python scripts/build_all.py --from features  # features + backtest only
    uv run python scripts/build_all.py --force          # rebuild the expensive steps too
                                                        # (re-downloads CDL, MODIS, POWER, NASS)

prices, features and backtest are cheap and always run when selected. cornmask, modis, hls_sample,
hls, weather and usda hit the network; they are skipped when their outputs exist unless --force.
hls needs EARTHDATA_TOKEN in .env; without a valid token the hls step is skipped (HLS features
are then absent and the H sets are skipped downstream) and the pipeline continues.
Ends by printing the headline metrics from reports/stageB_summary.csv.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling build_*.py scripts

from cornsat.config import DATA_PROCESSED, STATES, YEARS  # noqa: E402

STEPS = ["prices", "cornmask", "modis", "hls_sample", "hls", "weather", "usda", "features",
         "backtest"]


def _run_main(module: str, argv: list[str] | None = None):
    """Call ``scripts/<module>.py:main`` with a controlled sys.argv."""
    import importlib

    old = sys.argv
    sys.argv = [f"{module}.py", *(argv or [])]
    try:
        return importlib.import_module(module).main()
    finally:
        sys.argv = old


def step_prices(force: bool) -> None:
    _run_main("build_prices")


def step_cornmask(force: bool) -> None:
    from cornsat import cornmask

    years = cornmask.cdl_years()
    cached = all(cornmask._cache_path(s, y).exists() for s in STATES for y in years)
    if cached and not force:
        print(f"skip: all {len(STATES) * len(years)} corn-fraction rasters cached")
        return
    _run_main("build_cornmask", ["--refresh"] if force else [])


def step_modis(force: bool) -> None:
    from cornsat import modis

    if modis.OUT_PATH.exists() and not force:
        print(f"skip: {modis.OUT_PATH.name} exists")
        return
    if force:
        for y in YEARS:
            modis.build_year(y, STATES, refresh=True)
    _run_main("build_modis")


def step_hls_sample(force: bool) -> None:
    from cornsat import hlssample

    seasons = range(2013, 2027)
    cached = all(hlssample._mask_path(s, y).exists() for s in STATES for y in seasons)
    if cached and not force:
        print(f"skip: all {len(STATES) * len(seasons)} HLS mask caches present")
        return
    _run_main("build_hls_sample", ["--refresh"] if force else [])


def step_hls(force: bool) -> None:
    from cornsat import features, hls

    if features.HLS_STATE_PATH.exists() and not force:
        print(f"skip: {features.HLS_STATE_PATH.name} exists")
        return
    try:
        hls.token()
    except hls.EarthdataTokenError as exc:
        print(f"SKIP hls step: {exc}\n"
              "HLS features will be absent and H sets are skipped downstream.")
        return
    _run_main("build_hls", [])


def step_weather(force: bool) -> None:
    out = DATA_PROCESSED / "weather_daily.parquet"
    if out.exists() and not force:
        print(f"skip: {out.name} exists")
        return
    _run_main("build_weather")


def step_usda(force: bool) -> None:
    outs = [DATA_PROCESSED / f"usda_{n}.parquet" for n in ("condition", "progress", "yield")]
    if all(p.exists() for p in outs) and not force:
        print("skip: usda_*.parquet exist")
        return
    rc = _run_main("build_usda")
    if rc:
        raise SystemExit(rc)


def step_features(force: bool) -> None:
    _run_main("build_features")


def step_backtest(force: bool) -> None:
    _run_main("run_backtest")


def print_metrics() -> None:
    path = ROOT / "reports" / "stageB_summary.csv"
    if not path.exists():
        print("no reports/stageB_summary.csv")
        return
    s = pd.read_csv(path).set_index("strategy")
    cols = ["net_pnl", "sharpe", "hit_rate", "max_dd", "trades", "costs", "pnl_lo", "pnl_hi",
            "seasons_pos", "seasons"]
    pd.set_option("display.width", 200)
    print("\n== Final metrics: Stage B, 1 ZCZ, after costs, 2014-2026 (reports/stageB_summary.csv) ==")
    print(s[cols].sort_values("net_pnl", ascending=False).round(2).to_string())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0],
                   help="first step to run (default: prices)")
    p.add_argument("--force", action="store_true", help="rebuild expensive steps even if outputs exist")
    args = p.parse_args()

    todo = STEPS[STEPS.index(args.start):]
    print("steps:", " -> ".join(todo))
    for name in todo:
        print(f"\n#### step: {name}", flush=True)
        t0 = time.time()
        globals()[f"step_{name}"](args.force)
        print(f"#### {name} done in {time.time() - t0:.1f}s", flush=True)
    print_metrics()


if __name__ == "__main__":
    main()

"""Phase 8f: state-panel evaluation of satellite feature groups (S now, HLS "H" later).

Usage:
    uv run python scripts/run_state_panel.py                 # writes reports/state_panel_*.csv
    uv run python scripts/run_state_panel.py --out-dir /tmp/x --demean --source trading

Reads data/processed/{features_daily|features_trading,usda_yield}.parquet and the state
final-yield history 1990-2008 (NASS Quick Stats, cached to
data/interim/state_final_yields.parquet). Writes state_panel_accuracy.csv,
state_panel_paired.csv, state_panel_by_state.csv and state_panel_events.csv. Design:
module docstring of cornsat.panel.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import pandas as pd

from cornsat import panel as P
from cornsat.config import ROOT


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--source", choices=("daily", "trading"), default="daily")
    ap.add_argument("--demean", action="store_true", help="state demeaning inside training")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--asof", default=None, help="only finals released by this date")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    usda_yield = pd.read_parquet(P.USDA_YIELD_PATH)
    finals = P.state_final_yields(usda_yield, P.fetch_state_final_history(), asof=args.asof)
    panel = P.load_panel(args.source)
    seasons = sorted(panel["season"].unique())
    targets = P.state_targets(finals, seasons)
    data = P.build_dataset(panel, targets, P.usda_state_forecasts(usda_yield))
    groups = P.available_groups(panel)
    print(f"panel: {args.source}, seasons {seasons[0]}-{seasons[-1]}, groups available: {groups}")
    print(f"targets released: {int(targets['surprise'].notna().sum())} state-seasons "
          f"({targets.dropna(subset=['surprise'])['season'].min()}-"
          f"{targets.dropna(subset=['surprise'])['season'].max()})")

    _, acc, pair, bystate, skipped = P.compare(data, demean=args.demean, n_boot=args.n_boot)
    events = P.event_table(panel)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    acc.to_csv(out / "state_panel_accuracy.csv", index=False)
    pair.to_csv(out / "state_panel_paired.csv", index=False)
    bystate.to_csv(out / "state_panel_by_state.csv", index=False)
    events.to_csv(out / "state_panel_events.csv", index=False)
    print(f"wrote state_panel_{{accuracy,paired,by_state,events}}.csv to {out}")

    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
    fmt = {"rmse_A": 2, "rmse_B": 2, "mean_d": 2, "p_ttest": 3, "p_wilcoxon": 3,
           "boot_lo": 2, "boot_hi": 2}
    cols = ["window", "scheme", "A", "B", "n_seasons", "n_rows", "rmse_A", "rmse_B", "mean_d",
            "n_seasons_B_better", "p_ttest", "p_wilcoxon", "boot_lo", "boot_hi"]
    print("\nPaired comparisons, all snapshots pooled (d = se_A - se_B; > 0: B better)")
    print(pair[pair["snapshot"] == "ALL"][cols].round(fmt).to_string(index=False))
    print("\nRMSE by snapshot (bu/ac), W+C vs W+C+S and baselines")
    a = acc[acc["set"].isin(["trend", "usda", "W+C", "W+C+S", "W+C+H"])]
    print(a.pivot_table(index=["window", "scheme", "set"], columns="snapshot", values="rmse")
          .reindex(columns=[*P.SNAPSHOTS, "ALL"]).round(2).to_string())
    if len(skipped):
        print(f"\nskipped (missing group): {len(skipped)} window-pairs, e.g. "
              f"{skipped.iloc[0]['window']} {skipped.iloc[0]['A']} vs {skipped.iloc[0]['B']}")
    print("\nEvent timing (first prior-season z <= -1)")
    ecols = ["event", "state", "season", "feature", "first_cross", "z_at_cross",
             "below_at_event", "first_cross_after_event", "days_after_event", "min_z", "min_z_date"]
    e = events[[c for c in ecols if c in events]]
    print(e.round({c: 2 for c in e.select_dtypes("number")}).to_string(index=False))


if __name__ == "__main__":
    main()

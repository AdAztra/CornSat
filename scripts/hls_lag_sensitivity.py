"""HLS lag sensitivity of the state-panel comparison (W+C vs W+C+H, W+C+S vs W+C+H).

Usage:
    uv run python scripts/hls_lag_sensitivity.py                  # lags 3 7 10 14
    uv run python scripts/hls_lag_sensitivity.py --lags 7 14 --n-boot 500 --out-dir /tmp/x

Why: the panel makes an HLS composite usable on end_date + HLS_LAG_DAYS (7 d), but 4-7% of
the 2024-26 granules were produced more than 7 days after acquisition (p95 up to 14.7 d).
For each lag the daily as-of panel is rebuilt IN MEMORY from the cached inputs
(``features.load_inputs``; HLS from data/processed/hls_state*.parquet) with
``build_features(..., hls_spec=replace(HLS_SPEC, lag_days=lag))``. Nothing under
data/processed is written and no network is used (the state final-yield history comes from
its cache, data/interim/state_final_yields.parquet). Only the HLS block depends on the lag;
W, C and S are identical across lags.

Then ``panel.compare`` runs on the windows 2016+, 2019+ and 2022+ (LOSO and walk-forward;
walk-forward is skipped where it leaves fewer than MIN_TEST_SEASONS test seasons, i.e.
2022+), with the strict common sample of panel.WINDOWS, and fewer bootstrap draws.

Writes reports/hls_lag_sensitivity.csv: one row per lag x window x scheme x pair (all
snapshots pooled) with RMSE of each set, paired p-values (t, Wilcoxon), bootstrap CI and
n_seasons. Caveat: HLS before 2022 is an archive reconstruction (see run_state_panel.py).
"""

from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from cornsat import features as F
from cornsat import panel as P
from cornsat.config import ROOT

LAGS = (3, 7, 10, 14)
WINDOW_LABELS = ("2016+", "2019+", "2022+")
PAIRS = [("W+C", "W+C+H"), ("W+C+S", "W+C+H"), ("W+C", "W+C+S")]
SETS = ("W+C", "W+C+S", "W+C+H")


def state_panel(inputs: dict, lag: int) -> pd.DataFrame:
    """State rows of the daily as-of panel with HLS available on end_date + ``lag``
    (like ``panel.load_panel("daily")``, but built in memory)."""
    spec = replace(F.HLS_SPEC, lag_days=int(lag))
    p = F.build_features(inputs["modis"], inputs["weather"], inputs["usda"], hls=inputs["hls"],
                         hls_spec=spec)
    p = p[p["geo"].isin(P.STATE_LIST)].copy()
    p["date"] = pd.to_datetime(p["date"])
    p["season"] = p["season"].astype(int)
    return p.sort_values(["geo", "date"]).reset_index(drop=True)


def check_default_lag(panel7: pd.DataFrame) -> str:
    """Compare the in-memory lag-7 panel with the saved features_daily (H columns)."""
    if not P.FEATURES_DAILY_PATH.exists():
        return "features_daily.parquet absent: not checked"
    saved = P.load_panel("daily")
    cols = [c for c in P.GROUPS["H"] if c in saved]
    a = panel7.set_index(["geo", "date"])[cols]
    b = saved.set_index(["geo", "date"])[cols].reindex(a.index)
    same = np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True, rtol=0, atol=0)
    return "identical to features_daily" if same else "DIFFERS from features_daily"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--lags", type=int, nargs="+", default=list(LAGS))
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    inputs = F.load_inputs(presample=True)
    if inputs["hls"] is None:
        raise SystemExit("no hls_state.parquet: nothing to test")
    usda_yield = pd.read_parquet(P.USDA_YIELD_PATH)
    if not P.STATE_FINAL_CACHE.exists():
        raise SystemExit(f"{P.STATE_FINAL_CACHE} missing: run scripts/run_state_panel.py first")
    finals = P.state_final_yields(usda_yield, P.fetch_state_final_history())
    forecasts = P.usda_state_forecasts(usda_yield)
    windows = {k: P.WINDOWS[k] for k in WINDOW_LABELS}

    rows = []
    for lag in args.lags:
        panel = state_panel(inputs, lag)
        if lag == F.HLS_LAG_DAYS:
            print(f"lag {lag} (default): in-memory H columns {check_default_lag(panel)}")
        targets = P.state_targets(finals, sorted(panel["season"].unique()))
        data = P.build_dataset(panel, targets, forecasts)
        _, acc, pair, _, skipped = P.compare(data, windows=windows, pairs=PAIRS,
                                             n_boot=args.n_boot)
        a = acc[(acc["snapshot"] == "ALL") & acc["set"].isin(SETS)]
        rm = a.pivot_table(index=["window", "scheme"], columns="set", values="rmse")
        pr = pair[pair["snapshot"] == "ALL"]
        for _, r in pr.iterrows():
            rec = {"hls_lag_days": lag, "window": r["window"], "scheme": r["scheme"],
                   "A": r["A"], "B": r["B"], "n_seasons": r["n_seasons"], "n_rows": r["n_rows"],
                   "rmse_A": r["rmse_A"], "rmse_B": r["rmse_B"], "mean_d": r["mean_d"],
                   "n_seasons_B_better": r["n_seasons_B_better"], "p_ttest": r["p_ttest"],
                   "p_wilcoxon": r["p_wilcoxon"], "boot_lo": r["boot_lo"],
                   "boot_hi": r["boot_hi"]}
            for s in SETS:
                rec[f"rmse_{s}"] = rm.loc[(r["window"], r["scheme"]), s] if s in rm else np.nan
            rows.append(rec)
        sk = skipped[skipped["A"] == "*"] if len(skipped) else skipped
        for _, r in sk.iterrows():
            print(f"lag {lag}: skipped scheme in window {r['window']}: {r['reason']}")
        print(f"lag {lag}: done ({len(pr)} pooled comparisons)")

    out = pd.DataFrame(rows).sort_values(["window", "scheme", "A", "B", "hls_lag_days"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "hls_lag_sensitivity.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")

    pd.set_option("display.width", 220, "display.max_rows", 200)
    print("\nNOTE: H before 2022 is an archive reconstruction; 2022+ is the operational window "
          "(LOSO only).")
    print("\nAll snapshots pooled (d = se_A - se_B > 0: B better)")
    cols = ["window", "scheme", "A", "B", "hls_lag_days", "n_seasons", "n_rows", "rmse_A",
            "rmse_B", "mean_d", "p_ttest", "p_wilcoxon"]
    print(out[cols]
          .round({"rmse_A": 2, "rmse_B": 2, "mean_d": 2, "p_ttest": 3, "p_wilcoxon": 3})
          .to_string(index=False))


if __name__ == "__main__":
    main()

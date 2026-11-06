"""Phase 6 end to end: Stage A yield-surprise model + Stage B weekly ZCZ backtest.

Usage:
    uv run python scripts/run_backtest.py            # full run (bootstrap 10k, random 5k)
    uv run python scripts/run_backtest.py --quick    # fewer resamples, for a smoke run

Reads data/processed/{features_trading,usda_yield,zcz_front_dec}.parquet and the US
final-yield history 1990-2008 (NASS Quick Stats via usda.query, cached). Writes
data/processed/{model_predictions,backtest_trades}.parquet, reports/*.csv and
reports/figures/*.png. Prints the headline table. Design: reports/phase6_design.md.
"""

from __future__ import annotations

import argparse
import logging
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cornsat import backtest as B  # noqa: E402
from cornsat import model as M  # noqa: E402
from cornsat.config import DATA_PROCESSED, ROOT  # noqa: E402

REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
TRADES_PATH = DATA_PROCESSED / "backtest_trades.parquet"
SEASONS = range(M.FIRST_SEASON, 2027)
TRADE_SEASONS = range(2014, 2027)

# figure style (light surface, recessive axes; categorical slots 1-3 + neutral)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})  # fmt: skip


# --- Stage A --------------------------------------------------------------------------------
def stage_a(feat, targets, forecasts, targets_huber):
    preds = M.snapshot_predictions(feat, targets, forecasts)
    loso14 = preds[(preds["scheme"] == "loso") & (preds["season"] >= 2014)].assign(scheme="loso_2014+")
    acc = M.accuracy_table(pd.concat([preds, loso14], ignore_index=True))
    paired = pd.concat([M.paired_comparison(preds, scheme="wf"),
                        M.paired_comparison(preds, scheme="loso")], ignore_index=True)
    # sensitivity: fixed alpha = 10, and a Huber trend (walk-forward only)
    sens = []
    for label, tg, alphas in (("main (OLS trend, LOO alpha)", targets, M.ALPHAS),
                              ("fixed alpha=10", targets, (10.0,)),
                              ("Huber trend", targets_huber, M.ALPHAS)):
        p = preds if label.startswith("main") else M.snapshot_predictions(
            feat, tg, forecasts, schemes=("wf",), alphas=alphas)
        a = M.accuracy_table(p[p["scheme"] == "wf"])
        s = a.groupby("set", sort=False)["rmse"].mean().rename(label)
        sens.append(s)
    sens = pd.concat(sens, axis=1)
    sens.index.name = "set (mean WF RMSE over snapshots)"
    return preds, acc, paired, sens


# --- Stage B --------------------------------------------------------------------------------
def stage_b(feat, targets, forecasts, prices, n_boot, n_rand):
    td = {y: B.season_prices(prices, y).index for y in TRADE_SEASONS}
    schedule = B.build_schedule(td)
    signals, rb_preds = B.compute_signals(feat, targets, forecasts, schedule)
    positions = B.positions_table(signals)
    res = B.run_strategies(prices, positions)
    summary = B.summary_table(res)
    seasons = B.season_table(res)

    # robustness: bootstrap + random-sign tests
    rob = []
    short_w = res["always_short"][1]
    for n, (d, w) in res.items():
        if n == "flat":
            continue
        r = {"strategy": n}
        bs = B.block_bootstrap(w, n_boot, seed=0, other=short_w)
        r.update({k: bs[k] for k in ("pnl_lo", "pnl_hi", "p_pnl_le0", "sharpe_lo", "sharpe_hi")})
        r.update(diff_vs_short=bs["diff"], diff_vs_short_lo=bs["diff_lo"],
                 diff_vs_short_hi=bs["diff_hi"], p_not_beat_short=bs["p_diff_le0"])
        r.update(B.random_sign_tests(w, n_rand, seed=1))
        rob.append(r)
    rob = pd.DataFrame(rob).set_index("strategy")

    # satellite question: W+C+S vs W+C with the same rule; S vs flat
    sat = []
    for sig in B.SIGNALS:
        a, b = f"{sig}|W+C+S", f"{sig}|W+C"
        bs = B.block_bootstrap(res[a][1], n_boot, seed=2, other=res[b][1])
        sa, sb = seasons[a], seasons[b]
        sat.append({"signal": sig, "with_s": a, "without_s": b,
                    "pnl_with_s": summary.loc[a, "net_pnl"], "pnl_without_s": summary.loc[b, "net_pnl"],
                    "diff": bs["diff"], "diff_lo": bs["diff_lo"], "diff_hi": bs["diff_hi"],
                    "p_diff_le0": bs["p_diff_le0"], "seasons_s_better": int((sa > sb).sum()),
                    "seasons_s_worse": int((sa < sb).sum())})
        s_only = f"{sig}|S"
        bs = B.block_bootstrap(res[s_only][1], n_boot, seed=3)
        sat.append({"signal": sig, "with_s": s_only, "without_s": "flat",
                    "pnl_with_s": summary.loc[s_only, "net_pnl"], "pnl_without_s": 0.0,
                    "diff": summary.loc[s_only, "net_pnl"], "diff_lo": bs["pnl_lo"],
                    "diff_hi": bs["pnl_hi"], "p_diff_le0": bs["p_pnl_le0"],
                    "seasons_s_better": int((seasons[s_only] > 0).sum()),
                    "seasons_s_worse": int((seasons[s_only] < 0).sum())})
    sat = pd.DataFrame(sat)

    # sensitivity: thresholds, next-day execution, fixed alpha
    model_strats = [c for c in B.strategy_names(positions) if "|" in c]
    sens = {"main": summary.loc[model_strats + ["always_short"], "net_pnl"]}
    grids = {"level": [0, 1, 2, 3, 4], "change": [0, 0.25, 0.5, 1, 2], "vs_usda": [0, 1, 2, 3, 4]}
    thr_rows = []
    for sig, grid in grids.items():
        for t in grid:
            pos_t = B.positions_table(signals, thresholds={sig: t})
            names = [f"{sig}|{s}" for s in B.TRADE_SETS]
            r_t = B.run_strategies(prices, pos_t, names=names)
            for n in names:
                d, w = r_t[n]
                thr_rows.append({"strategy": n, "threshold": t, "net_pnl": w["net"].sum(),
                                 "sharpe": B.sharpe(w)})
    thr = pd.DataFrame(thr_rows)
    lag = B.run_strategies(prices, positions, exec_lag=1)
    sens["next_day_close"] = B.summary_table(lag)["net_pnl"].reindex(sens["main"].index)
    sig10, _ = B.compute_signals(feat, targets, forecasts, schedule, alphas=(10.0,))
    r10 = B.run_strategies(prices, B.positions_table(sig10), names=model_strats + ["always_short"])
    sens["fixed_alpha_10"] = B.summary_table(r10)["net_pnl"].reindex(sens["main"].index)
    sens = pd.DataFrame(sens)
    sens.index.name = "strategy"

    trades = pd.concat([w.assign(strategy=n) for n, (d, w) in res.items()], ignore_index=True)
    return dict(schedule=schedule, signals=signals, rb_preds=rb_preds, positions=positions,
                res=res, summary=summary, seasons=seasons, rob=rob, sat=sat, thr=thr,
                sens=sens, trades=trades)


# --- figures -----------------------------------------------------------------------------
def fig_pred_vs_actual(preds, best_set, path):
    snaps = ["Jul15", "Aug01", "Aug15", "Sep15"]
    p = preds[(preds["scheme"] == "wf") & preds["actual"].notna()]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True, sharey=True)
    lim = (-16, 16)
    for ax, sn in zip(axes, snaps):
        m = p[(p["snapshot"] == sn) & (p["set"] == best_set)]
        u = p[(p["snapshot"] == sn) & (p["set"] == "usda")].dropna(subset=["pred"])
        ax.plot(lim, lim, color=NEUTRAL, lw=1, ls="--", zorder=1)
        ax.axhline(0, color=GRID, lw=1, zorder=0)
        if len(u):
            ax.scatter(u["actual"], u["pred"], s=36, color=C2, edgecolor=SURF, lw=1.5,
                       zorder=2, label="USDA forecast − trend")
        ax.scatter(m["actual"], m["pred"], s=44, color=C1, edgecolor=SURF, lw=1.5, zorder=3,
                   label=f"model {best_set}")
        for _, r in m.iterrows():
            ax.annotate(str(r["season"])[2:], (r["actual"], r["pred"]), fontsize=7,
                        color=INK2, xytext=(3, 3), textcoords="offset points")
        e = m["pred"] - m["actual"]
        ax.set_title(f"{sn}  (RMSE {np.sqrt((e ** 2).mean()):.1f}, trend-only "
                     f"{np.sqrt((m['actual'] ** 2).mean()):.1f})", fontsize=9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("actual surprise (bu/ac)")
    axes[0].set_ylabel("predicted surprise (bu/ac)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"Walk-forward yield surprise, best set {best_set} (2014–2025)", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _equity(daily):
    d = daily.sort_values("date").reset_index(drop=True)
    return d["net"].cumsum().to_numpy(), d["season"].to_numpy()


def fig_equity(res, lines, path):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [C1, C2, C3, NEUTRAL]
    seas = None
    for (name, label), col in zip(lines, colors):
        eq, seas = _equity(res[name][0])
        ax.plot(np.arange(len(eq)), eq, color=col, label=label)
        ax.annotate(f"{label}  ${eq[-1]:,.0f}", (len(eq) - 1, eq[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=INK, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    starts = [0, *list(np.flatnonzero(np.diff(seas)) + 1)]
    ax.set_xticks(starts)
    ax.set_xticklabels([str(seas[i]) for i in starts])
    for s in starts[1:]:
        ax.axvline(s, color=GRID, lw=0.8)
    ax.set_ylabel("cumulative net P&L ($, 1 contract)")
    ax.set_xlim(0, len(seas) * 1.18)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Equity, Jun–Sep trading days only (walk-forward, after costs)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_season_bars(seasons, cols, path):
    s = seasons[[c for c, _ in cols]]
    x = np.arange(len(s))
    w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, ((c, label), col) in enumerate(zip(cols, [C1, C2, C3])):
        ax.bar(x + (i - (len(cols) - 1) / 2) * w, s[c], width=w * 0.92, color=col, label=label)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(s.index.astype(str))
    ax.set_ylabel("net P&L per season ($)")
    ax.legend(loc="upper right", fontsize=8, ncol=len(cols))
    ax.set_title("Net P&L by season (Jun–Sep, 1 ZCZ contract)", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def fig_2012(feat, path):
    f = feat[(feat["season"] == 2012) & (feat["date"].dt.month.between(5, 10))]
    panels = [("close", "ZCZ12 close (¢/bu)", C1), ("ge_anom", "good/excellent anomaly (pts)", C2),
              ("edd30_14d", "EDD>30°C, last 14 days", C2), ("ndvi_z", "NDVI z-score", C3)]
    fig, axes = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    # (date, label, y as axes fraction, side): June labels left of their line, August right
    marks = [(f.loc[f["edd30_14d"] > 20, "date"].min(), "EDD 14d > 20", 0.92, -1),
             (f.loc[f["ge_anom"] < -10, "date"].min(), "G/E anom < −10", 0.78, -1),
             (f.loc[f["ndvi_z"] < -1, "date"].min(), "NDVI z < −1", 0.25, 1),
             (f.loc[f["close"].idxmax(), "date"], "price peak", 0.10, 1)]
    for ax, (c, lab, col) in zip(axes, panels):
        ax.plot(f["date"], f[c], color=col)
        ax.set_ylabel(lab, fontsize=8)
        for d, txt, yf, side in marks:
            if pd.isna(d):
                continue
            ax.axvline(d, color=NEUTRAL, lw=1, ls="--")
            if c == "close":
                ax.annotate(f"{txt}: {d:%b %d}", (d, yf), xycoords=("data", "axes fraction"),
                            xytext=(4 * side, 0), textcoords="offset points", fontsize=7,
                            color=INK2, va="center", ha="left" if side > 0 else "right")
        if c in ("ge_anom", "ndvi_z"):
            ax.axhline(0, color=INK2, lw=0.8)
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b"))
    axes[0].set_title("2012: conditions and weather saw the drought weeks before the satellite",
                      loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- main --------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="1,000 bootstrap / random draws")
    args = ap.parse_args()
    n_boot, n_rand = (1000, 1000) if args.quick else (10_000, 5_000)
    logging.basicConfig(level=logging.WARNING)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    REPORTS.mkdir(exist_ok=True)
    FIGS.mkdir(exist_ok=True)

    us = M.load_us_yields()
    final = M.us_final_yields(us, M.fetch_final_history())
    targets = M.season_targets(final, SEASONS)
    targets_h = M.season_targets(final, SEASONS, method="huber")
    forecasts = M.us_forecasts(us)
    feat = M.prepare_features(pd.read_parquet(DATA_PROCESSED / "features_trading.parquet"))
    prices = pd.read_parquet(DATA_PROCESSED / "zcz_front_dec.parquet")

    # Stage A
    preds, acc, paired, sens_a = stage_a(feat, targets, forecasts, targets_h)
    targets.round(3).to_csv(REPORTS / "stageA_targets.csv")
    acc.round(4).to_csv(REPORTS / "stageA_accuracy.csv", index=False)
    paired.round(4).to_csv(REPORTS / "stageA_paired.csv", index=False)
    sens_a.round(3).to_csv(REPORTS / "stageA_sensitivity.csv")

    # Stage B
    b = stage_b(feat, targets, forecasts, prices, n_boot, n_rand)
    summ = b["summary"].join(b["rob"])
    summ.round(4).to_csv(REPORTS / "stageB_summary.csv")
    b["seasons"].round(1).to_csv(REPORTS / "stageB_season_pnl.csv")
    b["sat"].round(4).to_csv(REPORTS / "stageB_satellite.csv", index=False)
    b["sens"].round(1).to_csv(REPORTS / "stageB_sensitivity.csv")
    b["thr"].pivot_table(index="strategy", columns="threshold", values="net_pnl").round(1).to_csv(
        REPORTS / "stageB_threshold_sensitivity.csv")

    # processed outputs
    sp = preds.assign(kind="snapshot")
    rp = b["rb_preds"].assign(kind="rebalance", snapshot=None)
    rp["actual"] = rp["season"].map(targets["surprise"])
    rp = rp.merge(b["signals"][["season", "date", "usda"]], on=["season", "date"], how="left")
    allp = pd.concat([sp, rp], ignore_index=True)
    allp["train_seasons"] = allp["train_seasons"].apply(lambda v: [int(x) for x in v])
    allp.to_parquet(M.PREDICTIONS_PATH, index=False)
    b["trades"].to_parquet(TRADES_PATH, index=False)

    # figures
    wf = acc[acc["scheme"] == "wf"]
    mean_rmse = wf[wf["set"].isin(M.SET_NAMES)].groupby("set")["rmse"].mean()
    best_set = mean_rmse.idxmin()
    fig_pred_vs_actual(preds, best_set, FIGS / "pred_vs_actual_wf.png")
    model_strats = [c for c in b["summary"].index if "|" in c]
    ranked = b["summary"].loc[model_strats, "net_pnl"].sort_values(ascending=False)
    best = ranked.index[0]
    best_s = next(n for n in ranked.index if n.endswith("S"))
    lines = [("always_short", "always short"), (best, f"best model: {best}"),
             (best_s, f"best with S: {best_s}"), ("always_long", "always long")]
    fig_equity(b["res"], lines, FIGS / "equity_curves.png")
    fig_season_bars(b["seasons"], [("always_short", "always short"), (best, best),
                                   (best_s, best_s)], FIGS / "season_pnl.png")
    fig_2012(feat, FIGS / "timeline_2012.png")

    # print headline
    pd.set_option("display.width", 200)
    print("\n== Stage A: walk-forward RMSE (bu/ac), seasons 2014-2025 ==")
    piv = wf.pivot_table(index="set", columns="snapshot", values="rmse", sort=False)
    print(piv[list(M.SNAPSHOTS)].round(2).to_string())
    print("\n== Stage A: does S help? walk-forward paired squared errors (ALL = season mean) ==")
    pw = paired[(paired["scheme"] == "wf") & (paired["snapshot"] == "ALL")]
    print(pw[["base", "with_s", "n", "rmse_base", "rmse_with_s", "n_s_better",
              "p_ttest", "p_wilcoxon"]].round(3).to_string(index=False))
    cols = ["net_pnl", "sharpe", "hit_rate", "max_dd", "trades", "costs", "pnl_lo", "pnl_hi",
            "diff_vs_short", "p_not_beat_short", "p_random_full", "p_random_own"]
    print("\n== Stage B: walk-forward backtest 2014-2026, 1 ZCZ, after costs ==")
    print(summ[cols].sort_values("net_pnl", ascending=False).round(2).to_string())
    print("\n== Satellite vs no satellite (same rule) ==")
    print(b["sat"].round(3).to_string(index=False))
    print(f"\nbest Stage A set (mean WF RMSE): {best_set}; best strategy: {best}; "
          f"best with S: {best_s}")
    print(f"wrote {M.PREDICTIONS_PATH.name}, {TRADES_PATH.name}, reports/*.csv, reports/figures/*.png")


if __name__ == "__main__":
    main()

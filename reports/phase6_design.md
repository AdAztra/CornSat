# Phase 6 design: yield-surprise model and ZCZ backtest

Written before any model was fit. Everything below is fixed before results are seen.
Code: `src/cornsat/model.py` (Stage A), `src/cornsat/backtest.py` (Stage B), `scripts/run_backtest.py`.

**Target.** surprise(Y) = US final yield(Y) − trend(Y). The trend is an OLS linear fit of US final
yield on year over **1990 … Y−1** only. 1990–2008 come from NASS Quick Stats (`usda.query`, cached),
and 2009–2025 from `usda_yield.parquet`. Each season's target uses its own trend, so a training
target never contains information from its own season or later.

**Snapshots (Stage A).** The `features_trading` row on the first trading day on or after Jun 15,
Jul 1, Jul 15, Aug 1, Aug 15, Sep 1 and Sep 15. Training rows are the other seasons' rows at the
same snapshot.

**Feature sets** (US aggregate, unsuffixed columns):
- **W**: `edd30_cum`, `precip_30d_anom`, `gdd_cum_anom`. `edd30_cum` (extreme heat since Apr 1) replaces
  `edd30_jul`, which is identically 0 before Jul 1 and so is useless at the June snapshots and rebalances.
- **C**: `ge`, `ge_chg_2w`. The level `ge` replaces `ge_anom`, because `ge_anom` is NaN for 2009–2011:
  the USDA climatology needs 3 prior seasons. Every model compares seasons at the same date, so a
  per-date climatology would only shift the intercept, which ridge fits anyway.
- **S**: `ndvi_z`, `evi_anom_cum`, `ndvi_trend_anom`.
- **Combinations**: W+C (5 features), W+S (6), C+S (5), W+C+S (8).
- **Baselines**: *trend* (predict 0); *mean* (mean of training surprises); *USDA* (latest NASS
  forecast with available_date ≤ snapshot, minus trend; from the Aug 15 snapshot on).

**Model.** StandardScaler + ridge, both fit on training seasons only. Alpha is chosen by efficient
leave-one-out on the training seasons (`RidgeCV`) from {0.1, 0.3, 1, 3, 10, 30, 100, 300}. The model
has **no intercept**: the features are centred on their training means, which leaves the slopes
unchanged, and a large alpha shrinks the prediction toward **0, the trend**, not toward the mean of
5–16 noisy past surprises. The trend, fit on at least 19 years, is the prior, and "no signal" is a
possible outcome. The *mean* baseline shows what an intercept would add. Missing data:
features that are NaN in the target row are dropped from that prediction and the prediction is
flagged as a fallback. This is how 2025 S after Aug 10 falls back to the set without S. Training
seasons with NaN in a used feature are dropped. With fewer than 5 training seasons there is no
prediction.

**Validation.** (1) Leave-one-season-out over 2009–2025 (17 seasons). This trains on future seasons
and is reported for comparison only. (2) **Expanding walk-forward**: season Y is trained on seasons
2009 … Y−1 (at least 5), so predictions cover 2014–2026, with 12 scored seasons (2014–2025). Only (2)
feeds the backtest. Metrics are RMSE, MAE and correlation. The S question is answered by a
paired comparison of squared errors by season (W+C vs W+C+S; also W vs W+S and C vs C+S), using
the mean difference, a paired t-test, Wilcoxon, and the count of seasons where S helped.

**Trading.** One ZCZ contract of the campaign year (5,000 bu, so 1¢ = $50; no rolls in Jun–Sep).
Rebalance every Tuesday, or the next trading day if Tuesday is a holiday, from the first on or after
Jun 1 to the last on or before Sep 30. Trades are at that day's close using that day's
`features_trading` row. The position is flattened at the close of the last trading day ≤ Sep 30,
and is flat outside the window. Seasons 2014–2026 (13).
**Refit at every rebalance date:** the training row of each season y < Y is its own last trading
row with calendar day-of-year ≤ that of the rebalance date. The target uses surprises of y < Y only.

**Signals** (5 sets W, C, W+C, W+C+S, S × 3 rules = 15 strategies):
1. *Level*: position = −sign(pred) if |pred| > 2 bu/ac, else 0.
2. *Change*: position = −sign(Δpred) if |Δpred| > 0.5 bu/ac, else 0, where Δ is measured against the
   previous rebalance in the same season. The first rebalance is flat. 0.5 is set from the
   prediction scale (weekly Δ is much smaller than the level), not from P&L.
3. *Vs USDA*: once a NASS August-or-later forecast is available, position = −sign(pred −
   USDA surprise) if the gap is > 2 bu/ac, else 0. Flat before that.

Benchmarks: always short, always long, and flat over the same window.

**Costs.** Each contract traded costs $2.50 (half of the $5 round-trip commission) plus $12.50
(0.25¢ slippage), so $15 per contract per side, charged on |Δposition| at every change, including
entry and exit. A long-to-short flip costs $30.

**Metrics.** Total net P&L; P&L per season; hit rate, meaning the share of weeks in a position with
gross P&L > 0; Sharpe; max drawdown of the daily cumulative net P&L; number of position changes;
contracts traded; total costs. *Sharpe convention*: weekly net P&L ÷ notional, where notional is
the close at that season's first rebalance × $50. Sharpe is then mean/sd × √52 over all in-window
weeks, including flat weeks. Only about 17 weeks a year are traded, so a calendar-year Sharpe is
about √(17/52) ≈ 0.57 × this figure.

**Robustness.**
- Season-block bootstrap over the 13 seasons (10,000 draws): CI for total P&L and Sharpe, and for the
  paired difference against always-short and against the same strategy without S.
- Random-signal tests (5,000 draws, costs recomputed): (a) iid ±1 every week; (b) random signs on the
  strategy's own active weeks.
- Sensitivity: thresholds (level and vs USDA: 0, 1, 2, 3, 4; change: 0, 0.25, 0.5, 1, 2), and execution
  at the next day's close.

**Pre-declared questions.** (i) Does W+C+S beat W+C (Stage A errors; Stage B, same rule)? (ii) Does S
alone beat flat and random? (iii) Does anything beat always-short after costs? There are 15
strategies, so the best one is expected to look good by chance.

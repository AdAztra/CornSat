# Phase 6 results: yield-surprise model and ZCZ backtest

Design: `reports/phase6_design.md`. **The design note was committed together with these results,
so none of the analysis here is pre-registered: it is exploratory** (see Disclosures). Reproduce
with `uv run python scripts/run_backtest.py`. Tables are in `reports/*.csv` and figures in
`reports/figures/`. This version includes the audit corrections listed below.

## Verdict

**Negative.** The satellite features add no measurable yield accuracy, and no model strategy
beats a plain always-short Jun–Sep after costs.

- **Stage A, yield.** On the honest walk-forward (12 scored seasons, 2014–2025), no feature set
  beats the trend-only forecast on average over the snapshots (trend 6.4 bu/ac; best set C 6.7).
  The USDA forecast beats every model by a wide margin from mid-August (4.2 and 3.6 vs 6.2–8.1 at
  Aug 15 and 4.5–7.2 at Sep 15). Adding S to W+C makes the season-average error *worse* (RMSE 7.1 → 7.6; S
  helps in 5 of 12 seasons; p = 0.21 t-test, 0.38 Wilcoxon). The one snapshot where S looked
  helpful (Sep 15, 6.6 → 5.6, 8/12 seasons) is no longer significant after the audit fixes
  (p = 0.13 / 0.08; it was 0.03 / 0.003 before), and it comes after the USDA September report,
  which is better still.
- **Stage B, trading.** Of 15 model strategies the best is `vs_usda|C` (+4,190, Sharpe
  0.35, P&L CI −6,355 … 16,048), which compares the C-model with the latest USDA
  forecast and uses no satellite data. It is far below always-short (+24,035). **No model strategy has a
  P&L bootstrap CI that excludes zero on the positive side.** The pre-audit headline model
  strategy, `change|W` (+$10,882), was an artefact of the 1-day weather lag and of model churn:
  it is +1,105 now.
- **Satellite vs no satellite (trading).** W+C+S is far below always-short and not distinguishable from flat under any
  rule (level −2,578, change +1,892, vs_usda
  −400). The one comparison in which S looks better, `change|W+C+S` vs
  `change|W+C` (+17,925, CI 8,232 … 27,976,
  11 of 13 seasons), is a difference between a losing strategy (`change|W+C`
  −16,032, CI −28,005 … −4,525) and a flat one
  (`change|W+C+S` CI −7,488 … 12,118). S makes a
  losing signal roughly flat; it does not create a profitable one. With the level and vs_usda
  rules S is neutral to slightly worse. S alone is indistinguishable from flat or random.
- **Always-short** (+24,035, Sharpe 0.93, 10/13 seasons positive) is the only
  result that clears the season bootstrap (P(P&L ≤ 0) = 0.002) at a Bonferroni-style
  level; its random-sign p-values (0.015 full, 0.012 own) clear 5% but
  not 0.05/18. It is the known seasonal weather-premium decay, not something this project
  discovered, and 13 seasons is a short sample.

## Audit corrections

An independent audit found five issues. Each was fixed and has a test that fails on the old code.
Outputs were rebuilt (prices, weather, USDA, features, backtest) with the existing scripts.

1. **Weather timing (HIGH).** `weather.add_features` set `available_date = date + 1`; NASA POWER
   really lags about 3 days (on 2026-11-06 the series ended on 2026-11-03). It is now the module
   constant `WEATHER_LAG_DAYS = 3` and applies to the main and the pre-sample weather. In
   `features.py` no 1-day assumption was hard-coded beyond docs; the weather staleness limit was
   7 days measured from the observation date, which after the lag change would have left only 4
   days of tolerated outage instead of 6, so it is now `WEATHER_LAG_DAYS + 6` (9 days).
   *Effect:* weather features are a few days staler, so W-based results move a lot (table
   below); C-only and S-only results are unchanged. Stage A mean RMSE: W 7.72 → 8.56, W+S 7.14 →
   8.08; Jun 15 W 7.2 → 12.6 (one prediction, 2018: −31 bu/ac, extrapolation from a 9-season fit
   with EDD-cum 15.3 against typical values below 5 in early June).
2. **Change rule traded on model churn (MEDIUM).** The "change" signal compared the prediction at
   rebalance d with the prediction stored at the previous rebalance, although the model is
   refit each time (new LOO alpha, feature-set fallback, training seasons). Now
   Δ = model_d(features at d) − model_d(features at the previous rebalance of the same season),
   one fitted model scoring two rows (no look-ahead: both rows are dated ≤ d). Δ is NaN, so the
   position is flat, on a season's first rebalance and whenever the earlier row lacks a column
   the model at d uses. `model.predict_one(..., x_prev=)` returns `delta`;
   `backtest.change_position(delta, thr)` replaced the old prediction-diff signature.
   Position 0 means flat, as for the other rules (below threshold, flat).
   *Effect:* the change strategies move by up to ±$22k (table below). The old numbers were
   largely churn.
3. **US aggregate composition (LOW).** New columns `us_n_states_modis` (states with `ndvi_anom`)
   and `us_n_states_weather` (states with `gdd_cum_anom`) on the US rows and in
   `features_trading`; the aggregation is unchanged. *Effect:* none on any number. In 2009–2026
   the MODIS aggregate uses fewer than 5 states on 160 days, all in April and May (129 in April, 31 in May),
   so every Jun–Sep trading row uses either all 5 states or none (36 rows, the 2025 gap). The
   weather aggregate always uses all 5 or none.
4. **USDA 2009–2011 forecast release dates (LOW).** Those rows carry a bulk load_time
   (2012-01-01), so the rule date was used and could precede the true release. Added
   `usda.RELEASE_CALENDAR` (Crop Production report days for Aug–Nov 2009–2011; entered from
   memory and flagged in the code as **to be verified**); forecast `available_date` is
   max(rule date, next business day after the calendar date, next business day after a valid
   load_time). *Effect:* 5 of 12 availability dates moved later (2009 Aug 11→13, Sep 11→14;
   2010 Aug 11→13; 2011 Aug 11→12, Oct 12→13). No number changes: only 2014+ forecasts enter the
   scored comparisons.
5. **Partial price bar (NOTE).** The last ZCZ26 row (2026-11-05) was an intraday snapshot dated on
   the download date. `prices.read_contract` now drops rows dated on or after the download date
   (from the file name `…-MM-DD-YYYY.csv`, else the footer). *Effect:* the last ZCZ26 row is
   2026-11-04. No number changes: the feature panel stops on Oct 31 and the traded window ends
   Sep 30.

**Net P&L by strategy, $ (Sharpe in brackets), cumulative effect of the fixes.** Fixes 3–5 do not
change a number, so the last column is the final result. The original column reproduces the
pre-audit report exactly.

| strategy | original (1-day lag, prediction-diff change rule) | after fix 1 (3-day lag) | after fix 2 (same-model change; = final, fixes 3-5 change no number) |
|---|---|---|---|
| always_short | +24,035 (Sh 0.93) | +24,035 (Sh 0.93) | +24,035 (Sh 0.93) |
| vs_usda\|C | +4,190 (Sh 0.35) | +4,190 (Sh 0.35) | +4,190 (Sh 0.35) |
| change\|W | +10,882 (Sh 0.65) | −9,568 (Sh −0.41) | +1,105 (Sh 0.01) |
| change\|W+C | +4,288 (Sh 0.25) | −1,305 (Sh 0.03) | −16,032 (Sh −0.86) |
| change\|W+C+S | −7,970 (Sh −0.40) | −13,648 (Sh −0.58) | +1,892 (Sh 0.10) |
| change\|C | +3,415 (Sh 0.24) | +3,415 (Sh 0.24) | −20,712 (Sh −1.21) |
| change\|S | +2,272 (Sh 0.18) | +2,272 (Sh 0.18) | +492 (Sh 0.09) |
| level\|W | +1,458 (Sh 0.06) | +1,158 (Sh 0.04) | +1,158 (Sh 0.04) |
| level\|W+C | −7,470 (Sh −0.41) | −2,225 (Sh −0.05) | −2,225 (Sh −0.05) |
| level\|W+C+S | −3,980 (Sh −0.31) | −2,578 (Sh −0.17) | −2,578 (Sh −0.17) |
| vs_usda\|W+C+S | −645 (Sh −0.22) | −400 (Sh −0.18) | −400 (Sh −0.18) |

**Stage A, mean walk-forward RMSE over the 7 snapshots (bu/ac).** Fix 2 does not touch Stage A.
Fixes 3–5 change nothing here.

| | W | C | S | W+C | W+S | C+S | W+C+S |
|---|---|---|---|---|---|---|---|
| original | 7.72 | 6.73 | 7.41 | 6.99 | 7.14 | 7.56 | 7.46 |
| after fix 1 (final) | 8.56 | 6.73 | 7.41 | 6.98 | 8.08 | 7.56 | 7.43 |

**Headline before → after.**
- always-short: +24,035 → +24,035, Sharpe 0.93 → 0.93; bootstrap P(≤0) 0.002 → 0.002, random-sign p 0.015 → 0.015 (unchanged: it uses no model).
- best model strategy: `change|W` +10,882, Sharpe 0.65, random-sign p 0.13 / 0.05 (full / own) → `vs_usda|C` +4,190, Sharpe 0.35, p 0.27 / 0.14. `change|W` itself is now +1,105 (Sharpe 0.01); with a 3-day lag but the old churn rule it was −9,568.
- `change|W+C+S` vs `change|W+C`: −12,258 (CI −24,315 … −290, S worse) → +17,925 (CI 8,232 … 27,976, S better). The sign flipped: the old W+C figure (+4,288) was a churn artefact.
- `level|W+C+S` vs `level|W+C`: +3,490 → −352. `vs_usda|W+C+S` vs `vs_usda|W+C`: −58 → −1,150.

## Stage A: yield surprise (bu/ac)

Target: final − OLS trend on 1990..Y−1 (`stageA_targets.csv`). Ridge without an intercept on
standardised features (training seasons only), with alpha chosen by LOO inside training.

**Walk-forward RMSE, seasons 2014–2025 (n = 12; S alone n = 11 from Aug 15 because of the 2025
gap).** This is the honest number. Bold is the best model set per snapshot.

| set | Jun15 | Jul01 | Jul15 | Aug01 | Aug15 | Sep01 | Sep15 |
|---|---|---|---|---|---|---|---|
| W | 12.6 | 10.4 | 8.3 | 7.6 | 7.6 | 6.2 | 7.1 |
| C | 6.2 | 11.1 | **5.5** | 7.4 | **6.2** | 5.3 | 5.4 |
| S | 7.0 | 10.4 | 6.8 | 9.4 | 6.7 | 6.8 | 4.8 |
| W+C | 6.6 | **9.3** | 6.7 | 8.1 | 6.7 | **4.8** | 6.6 |
| W+S | 14.4 | 10.0 | 8.5 | **6.7** | 6.5 | 5.6 | 4.8 |
| C+S | **6.1** | 11.8 | 6.4 | 10.2 | 7.4 | 6.5 | **4.5** |
| W+C+S | 6.6 | 10.1 | 6.9 | 9.7 | 8.1 | 5.0 | 5.6 |
| trend (predict 0) | 6.4 | 6.4 | 6.4 | 6.4 | 6.4 | 6.4 | 6.4 |
| mean of past surprises | 9.7 | 9.7 | 9.7 | 9.7 | 9.7 | 9.7 | 9.7 |
| **USDA forecast − trend** |  |  |  |  | **4.2** | **4.2** | **3.6** |

- **Jun 15 W (12.6) and W+S (14.5)** are driven by one prediction, 2018 (−31 and −40 bu/ac against
  an actual of +4.8), from a nine-season fit on a feature (`edd30_cum`) that is close to 0 in
  most early Junes. It is an instability of the small-sample ridge, not a signal.
- **Correlations.** Walk-forward correlations with the actual surprise are 0.26–0.62 for C, mixed in sign for W and S before September, and 0.74–0.83 for USDA.
- **LOSO** (trains on future seasons, 17 seasons) gives a trend RMSE of 11.6, driven by 2012's
  −38 bu/ac. Restricted to the same 2014–2025 seasons, LOSO looks much better than walk-forward
  (for example W+C at Sep 1 scores 3.3 vs 4.8, C at Jul 15 3.9 vs 5.5). The usual LOSO numbers in
  this literature are flattered by training on later seasons.

**Does S help? Walk-forward paired squared errors** (d = e²(base) − e²(with S); ALL = each
season's mean over the 7 snapshots, so n is still 12).

| base | with S | n | RMSE base | RMSE with S | seasons S better | p (t) | p (Wilcoxon) |
|---|---|---|---|---|---|---|---|
| W+C | W+C+S | 12 | 7.1 | 7.6 | 5 | 0.21 | 0.38 |
| W | W+S | 12 | 8.8 | 8.6 | 5 | 0.80 | 0.97 |
| C | C+S | 12 | 7.0 | 7.9 | 5 | 0.11 | 0.23 |

- **By snapshot, W+C vs W+C+S.** S is better at Sep 15 (6.6 → 5.6, 8/12 seasons, p = 0.13 t /
  0.08 Wilcoxon, not significant). It is worse or tied at the other six snapshots, with one
  nominal p < 0.05 in the wrong direction (Jul 1, Wilcoxon p = 0.04, S better in 2/12). With 21
  snapshot × pair tests such values are what chance produces.
- **Late-season S signal.** S alone at Sep 15 (RMSE 4.8, r = 0.71) is a consistent late-season
  signal, but the USDA September forecast available on the same date has an RMSE of 3.6.
- **LOSO agrees.** Season-average ALL: W+C 10.2 vs W+C+S 11.0; C 9.3 vs C+S 10.8.
- **How strong is this evidence?** With n = 12 a paired test only detects large effects. The
  data cannot rule out a gain of about 1 bu/ac RMSE, but they show no sign of one.

**Sensitivity** (mean walk-forward RMSE over snapshots, `stageA_sensitivity.csv`):

| | W | C | S | W+C | W+S | C+S | W+C+S | trend | USDA |
|---|---|---|---|---|---|---|---|---|---|
| main (OLS trend, LOO alpha) | 8.6 | 6.7 | 7.4 | 7.0 | 8.1 | 7.6 | 7.4 | 6.4 | 4.0 |
| fixed alpha=10 | 7.2 | 5.6 | 7.0 | 6.2 | 7.0 | 6.1 | 6.4 | 6.4 | 4.0 |
| Huber trend | 7.2 | 6.5 | 6.6 | 6.0 | 6.8 | 6.9 | 6.2 | 5.8 | 4.0 |

- **More shrinkage helps.** A fixed alpha = 10 lets C and W+C edge past the trend. LOO often
  picks alpha = 0.1, close to OLS, with only 5–16 seasons.
- **The S conclusion holds.** S still does not improve on the matching set without S.

![walk-forward predicted vs actual](figures/pred_vs_actual_wf.png)

## Stage B: weekly ZCZ backtest, walk-forward, 2014–2026 (13 seasons, 225 weeks)

Each week is one ZCZ contract, rebalanced on Tuesday at the close and flat outside Jun–Sep. Costs
are $15 per contract per side, so a flip costs $30. Signals: **level** = −sign(prediction) when
|prediction| > 2 bu/ac; **change** = −sign(Δ) when |Δ| > 0.5, with Δ the same-model change
described under fix 2; **vs_usda** = −sign(prediction − USDA forecast surprise) when the gap
exceeds 2. Sharpe uses weekly P&L ÷ (first-rebalance close × $50), then × √52 over in-window
weeks; multiply by about 0.57 for a calendar-year figure. The CI is a 95% season-block bootstrap
(10,000 draws).
- **p(¬>short)**: bootstrap probability that the strategy does not beat always-short.
- **p rand**: share of 5,000 random-sign runs with net P&L ≥ the strategy's. *full* means iid ±1
  every week. *own* means random signs on the strategy's own active weeks.

| strategy | net P&L $ | Sharpe | hit | max DD $ | trades | costs $ | seasons + | P&L 95% CI | Sharpe 95% CI | vs short $ | p(¬>short) | p rand full | p rand own |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **always_short** | **24,035** | **0.93** | 0.56 | 10,080 | 26 | 390 | 10 | 7,885 … 40,873 | 0.21 … 1.80 | 0 | – | 0.02 | 0.01 |
| vs_usda\|C | 4,190 | 0.35 | 0.54 | 3,715 | 40 | 660 | 7 | −6,355 … 16,048 | −0.67 … 1.27 | −19,845 | 0.96 | 0.27 | 0.14 |
| change\|W+C+S | 1,892 | 0.10 | 0.49 | 6,520 | 123 | 2,520 | 6 | −7,488 … 12,118 | −0.47 … 0.78 | −22,142 | 0.99 | 0.33 | 0.29 |
| level\|W | 1,158 | 0.04 | 0.50 | 6,688 | 59 | 930 | 6 | −12,945 … 15,955 | −0.82 … 0.93 | −22,878 | 0.98 | 0.35 | 0.33 |
| change\|W | 1,105 | 0.01 | 0.53 | 9,805 | 113 | 2,070 | 6 | −14,668 … 14,893 | −0.79 … 0.88 | −22,930 | 0.97 | 0.35 | 0.34 |
| vs_usda\|W | 780 | 0.10 | 0.46 | 4,225 | 38 | 570 | 6 | −7,380 … 9,653 | −0.68 … 0.94 | −23,255 | 0.98 | 0.37 | 0.34 |
| vs_usda\|W+C | 750 | −0.02 | 0.44 | 4,738 | 47 | 750 | 6 | −8,835 … 10,646 | −0.98 … 0.89 | −23,285 | 0.98 | 0.37 | 0.33 |
| change\|S | 492 | 0.09 | 0.50 | 4,608 | 87 | 1,470 | 7 | −10,338 … 11,303 | −0.89 … 1.00 | −23,542 | 0.98 | 0.38 | 0.35 |
| flat | 0 | – | – | 0 | 0 | 0 | 0 | | | | | | |
| vs_usda\|W+C+S | −400 | −0.18 | 0.39 | 5,242 | 36 | 600 | 5 | −11,578 … 13,303 | -1.19 … 0.87 | −24,435 | 0.98 | 0.40 | 0.44 |
| level\|S | −828 | −0.09 | 0.56 | 6,468 | 74 | 1,140 | 7 | −10,740 … 8,068 | −0.88 … 0.84 | −24,862 | 0.99 | 0.42 | 0.44 |
| level\|W+C | −2,225 | −0.05 | 0.52 | 10,772 | 80 | 1,350 | 8 | −22,752 … 12,185 | −0.92 … 0.77 | −26,260 | 0.98 | 0.46 | 0.49 |
| level\|W+C+S | −2,578 | −0.17 | 0.52 | 11,228 | 90 | 1,440 | 7 | −26,084 … 17,775 | -1.19 … 0.99 | −26,612 | 0.96 | 0.47 | 0.50 |
| vs_usda\|S | −5,442 | −0.40 | 0.46 | 7,960 | 37 | 630 | 4 | −14,423 … 3,363 | -1.24 … 0.48 | −29,478 | 1.00 | 0.56 | 0.75 |
| level\|C | −8,232 | −0.40 | 0.50 | 13,702 | 64 | 1,020 | 5 | −24,950 … 5,450 | -1.16 … 0.35 | −32,268 | 1.00 | 0.64 | 0.73 |
| change\|W+C | −16,032 | −0.86 | 0.42 | 18,862 | 126 | 2,520 | 2 | −28,005 … −4,525 | -1.36 … −0.26 | −40,068 | 1.00 | 0.84 | 0.92 |
| change\|C | −20,712 | −1.21 | 0.40 | 22,860 | 132 | 2,400 | 3 | −39,185 … −7,932 | -1.81 … −0.59 | −44,748 | 1.00 | 0.91 | 0.98 |
| always_long | −24,815 | −0.97 | 0.44 | 28,498 | 26 | 390 | 3 | −41,653 … −8,665 | -1.84 … −0.25 | −48,850 | 1.00 | 0.95 | 0.96 |

**Satellite vs the same rule without S** (`stageB_satellite.csv`; difference CI by paired season
bootstrap; bold = CI excludes 0):

| with S | without S | P&L with S $ | P&L without S $ | diff $ | 95% CI | seasons S better / worse |
|---|---|---|---|---|---|---|
| level\|W+C+S | level\|W+C | −2,578 | −2,225 | −352 | −13,783 … 14,461 | 4 / 7 |
| level\|S | flat | −828 | 0 | −828 | −10,755 … 8,123 | 7 / 5 |
| change\|W+C+S | change\|W+C | 1,892 | −16,032 | **+17,925** | 8,232 … 27,976 | 11 / 2 |
| change\|S | flat | 492 | 0 | +492 | −10,550 … 11,383 | 7 / 5 |
| vs_usda\|W+C+S | vs_usda\|W+C | −400 | 750 | −1,150 | −8,663 … 6,133 | 4 / 6 |
| vs_usda\|S | flat | −5,442 | 0 | −5,442 | −14,376 … 3,218 | 4 / 8 |

**Net P&L by season, $** (selected strategies; all are in `stageB_season_pnl.csv`):

| season | always short | vs_usda\|C | change\|W+C | change\|W+C+S | change\|C | level\|W+C | level\|W+C+S | vs_usda\|W+C | vs_usda\|W+C+S |
|---|---|---|---|---|---|---|---|---|---|
| 2014 | 6,645 | −2,605 | 1,460 | 1,180 | 332 | 678 | 2,230 | −2,605 | −2,605 |
| 2015 | −618 | 1,535 | 1,130 | 3,022 | −1,685 | 2,852 | 2,510 | −42 | 632 |
| 2016 | 4,795 | 40 | −3,260 | −1,418 | 482 | 558 | −1,360 | −610 | −948 |
| 2017 | 1,995 | 532 | −1,468 | 28 | −1,550 | −718 | −2,272 | 702 | 132 |
| 2018 | 2,358 | 1,010 | −668 | −400 | −1,840 | −898 | 1,490 | −1,042 | −1,042 |
| 2019 | 2,770 | −692 | −790 | 1,595 | −1,892 | 380 | −3,520 | −760 | −1,272 |
| 2020 | −2,080 | 1,120 | −1,440 | −1,428 | −1,928 | −2 | 28 | 378 | 1,120 |
| 2021 | 1,982 | 302 | −1,598 | 2,265 | −1,938 | 692 | 230 | −240 | −580 |
| 2022 | 1,795 | 4,565 | −172 | 122 | −542 | 272 | 4,920 | 3,332 | 5,290 |
| 2023 | 3,182 | −592 | −4,825 | −1,300 | −8,898 | −8,195 | −8,195 | −1,085 | −190 |
| 2024 | 1,832 | −610 | −218 | −560 | −670 | 110 | −225 | 535 | −748 |
| 2025 | 1,120 | −330 | −1,265 | −910 | −1,122 | −285 | −285 | 478 | 478 |
| 2026 | −1,742 | −85 | −2,920 | −305 | 538 | 2,330 | 1,872 | 1,710 | −668 |

### Robustness

- **Multiple testing.** There are 18 strategies (15 model strategies and 3 benchmarks). No model
  strategy has a P&L bootstrap CI that excludes 0 on the positive side. Two have one that
  excludes 0 on the negative side (`change|C` −20,712, `change|W+C` −16,032); flipping their
  sign after seeing this would be data snooping and is not proposed. The satellite comparison
  `change|W+C+S` vs `change|W+C` has a CI that excludes 0, but it is the difference between a
  losing and a flat strategy (see the verdict), it is one of six comparisons, and the S model
  has no better yield accuracy in Stage A.
- **Sensitivity** (`stageB_sensitivity.csv`, `stageB_threshold_sensitivity.csv`). Model-strategy
  P&L is unstable.
  - *Fixed alpha = 10*: `change|W` goes from +1,105 to −8,665, `change|S` from +492 to +13,362,
    `level|C` from −8,232 to +8,285, and `level|W+C` from −2,225 to +4,875.
  - *Next-day close*: `change|S` goes from +492 to −4,545, `change|W+C+S` from +1,892 to +6,410, and `level|W+C` from −2,225 to −6,658.
  - *Thresholds*: `change|S` ranges from −478 to +11,638, `level|W` from −11,895 to +5,532, and `change|W+C` from −16,032 to −2,442, across thresholds.
  - *Always-short* is stable: $23,072 at next-day close.
  - A P&L that swings by ±$10–20k with a nuisance parameter is noise, not signal.
- **The change rule.** After fix 2 the change signal measures how much the same model's view
  moves as the features move, not how much the model itself was refit. It still depends on the
  weekly feature changes in `ge`, weather and NDVI, which are mostly noise and are priced by
  the market within days; the C-based change rules lose money (`change|C` −20,712, 132 trades).

![equity](figures/equity_curves.png)
![per-season P&L](figures/season_pnl.png)

### Why satellite data arrives too late (2012)

2012 is a training season and is not traded, but it shows the mechanism.
- Extreme heat (14-day EDD above 20) appeared on **Jun 27** (with the 3-day weather lag; Jun 25
  before the audit fix), and the G/E rating was more than 10 points below normal by **Jun 26**.
  ZCZ12 had already started rallying in mid-June.
- US `ndvi_z` only fell below −1 on **Aug 9**, after a 14-day publication lag and with
  16-day composites. The price peaked on **Aug 21**.
- By the time the satellite confirms damage, conditions, weather and the market have already
  priced it.

![2012 timeline](figures/timeline_2012.png)

## Disclosures

- **Exploratory analysis.** The design note (`phase6_design.md`) was committed together with the
  results, not before them, so the design is not a pre-registration and the analysis is
  exploratory. Choices such as thresholds, sets and rules were written down by the same people
  who then saw the data.
- **The S feature mix was not pre-registered.** The three satellite features (`ndvi_z`,
  `evi_anom_cum`, `ndvi_trend_anom`) were chosen without a committed specification.
- **Multiple comparisons.** About 18 strategies were tested (15 model strategies plus 3
  benchmarks), plus 21 snapshot × pair tests in Stage A. Read any single significant result with
  a Bonferroni-style correction (0.05/18 = 0.003 for a strategy); on that scale only the
  always-short bootstrap passes, and no model result does.
- **Final yields.** Stored FINAL yields are later NASS revisions, not the first January release.
  The difference is at most about 0.7 bu/ac and is negligible for these results.
- **Release calendar from memory.** `usda.RELEASE_CALENDAR` (2009–2011 Crop Production days) was
  entered from memory and should be verified against the NASS archive. It affects no reported
  number.

## Caveats

- **Small sample.** There are 12 scored walk-forward seasons (2014–2025) and 13 trading seasons
  (2014–2026). The walk-forward starts with 5 training seasons, and 2012 (−38 bu/ac) dominates
  the early fits. The power to detect anything short of a large effect is low.
- **Reprocessed MODIS (C6.1).** Today's archive is not exactly what was available in real time.
  The 14-day lag is an approximation, and the 2003–2008 climatology uses the CDL-2008 mask.
- **Reanalysis weather.** NASA POWER is gridded reanalysis at 35 points, not real-time station
  data. Daily values are assumed available 3 days after the date (`WEATHER_LAG_DAYS`); the
  stored values are today's vintage and the latest days may have been revised.
- **2025 satellite gap.** MODIS is NaN after Aug 10, 2025. S strategies are flat for the rest of
  2025 (7 rebalances). Combined sets fall back to the set without S, and these predictions are
  flagged `s_dropped` in `model_predictions.parquet`.
- **2026 is partial.** It has no final yield, so it is not scored in Stage A. It is traded in
  Stage B, because Jun–Sep 2026 prices are complete.
- **Feature substitutions.** `ge` replaces `ge_anom`, because `ge_anom` is NaN in 2009–2011.
  `edd30_cum` replaces `edd30_jul`, which is 0 before July. Early-June predictions often drop
  `ge_chg_2w` or `evi_anom_cum` because those are not yet reported.
- **Trading assumptions.** Each trade is one contract at the settlement close. There is no
  margin or interest on collateral, and liquidity or limit-move days are ignored.

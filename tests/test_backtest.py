"""Phase 6 Stage B tests: P&L math and costs, trading window, rebalance calendar,
signal rules and the no-look-ahead property of positions."""

import numpy as np
import pandas as pd
import pytest

from cornsat import backtest as B
from cornsat import model as M
from test_model import synth_features


def prices_frame(close: pd.Series, season: int) -> pd.DataFrame:
    return pd.DataFrame({"date": close.index, "campaign": season, "close": close.to_numpy()})


# --- P&L math ----------------------------------------------------------------------------------
def test_pnl_hand_example():
    d = pd.to_datetime(["2020-06-02", "2020-06-03", "2020-06-04", "2020-06-05", "2020-06-08"])
    close = pd.Series([100.0, 102.0, 101.0, 105.0, 104.0], index=d)
    daily, weekly = B.simulate_season(close, [d[0], d[2]], [1, -1], d[4])
    # long 1 from d0: +2c, -1c; flip to short at d2 (2 contracts); short: +4c (loss), -1c (gain)
    assert daily["pos"].tolist() == [1, 1, -1, -1, 0]
    assert daily["gross"].tolist() == [0, 100, -50, -200, 50]
    assert daily["cost"].tolist() == [15, 0, 30, 0, 15]
    assert daily["net"].sum() == pytest.approx(-100 - 60)
    assert weekly["pos"].tolist() == [1, -1]
    assert weekly["gross"].tolist() == [50, -150]
    assert weekly["cost"].tolist() == [15, 45]
    assert weekly["net"].sum() == pytest.approx(daily["net"].sum())
    assert weekly["d_close"].tolist() == [1.0, 3.0]


def test_costs_only_on_position_changes():
    d = pd.bdate_range("2020-06-01", periods=10)
    close = pd.Series(np.linspace(400, 409, 10), index=d)
    _, w_hold = B.simulate_season(close, d[[0, 2, 4, 6]], [-1, -1, -1, -1], d[9])
    assert w_hold["cost"].sum() == pytest.approx(2 * B.COST_PER_SIDE)  # entry + exit only
    assert w_hold["gross"].sum() == pytest.approx(-9 * 50)
    _, w_flat = B.simulate_season(close, d[[0, 2, 4, 6]], [0, 0, 0, 0], d[9])
    assert w_flat["cost"].sum() == 0 and w_flat["gross"].sum() == 0
    _, w_flip = B.simulate_season(close, d[[0, 2, 4, 6]], [1, -1, 0, 1], d[9])
    assert w_flip["cost"].sum() == pytest.approx((1 + 2 + 1 + 1 + 1) * B.COST_PER_SIDE)
    assert B.COST_PER_SIDE == pytest.approx(5 / 2 + 0.25 * 50)


def test_flat_outside_window_and_calendar():
    season = 2021
    d = pd.bdate_range("2021-04-01", "2021-11-30")
    d = d[d != pd.Timestamp("2021-07-06")]  # a "holiday" Tuesday -> Wednesday
    close = pd.Series(500 + np.arange(len(d), dtype=float), index=d)
    prices = prices_frame(close, season)
    reb = B.rebalance_dates(d, season)
    assert reb[0] == pd.Timestamp("2021-06-01")
    assert pd.Timestamp("2021-07-07") in reb and pd.Timestamp("2021-07-06") not in reb
    assert all(r.day_of_week == 1 for r in reb if r != pd.Timestamp("2021-07-07"))
    ex = B.exit_date(d, season)
    assert ex == pd.Timestamp("2021-09-30") and max(reb) < ex
    sch = pd.DataFrame({"season": season, "date": reb, "position": 1.0})
    daily, weekly = B.simulate(prices, sch)
    assert daily["date"].min() == reb[0] and daily["date"].max() == ex
    assert daily.loc[daily["date"] == ex, "pos"].item() == 0
    # always long on a +1c/day series: P&L = (close[exit] - close[first]) * 50 - 2 * cost
    expect = (close[ex] - close[reb[0]]) * 50 - 2 * B.COST_PER_SIDE
    assert weekly["net"].sum() == pytest.approx(expect)
    assert (weekly["notional"] == close[reb[0]] * 50).all()
    # next-day execution: every trade one trading day later, exit unchanged
    daily1, weekly1 = B.simulate(prices, sch, exec_lag=1)
    assert weekly1["start"].iloc[0] == pd.Timestamp("2021-06-02")
    assert daily1["date"].max() == ex


def test_sep30_tuesday_rebalance_dropped():
    d = pd.bdate_range("2025-05-01", "2025-10-31")  # Sep 30 2025 is a Tuesday
    reb = B.rebalance_dates(d, 2025)
    assert reb[-1] == pd.Timestamp("2025-09-23")


# --- signals -----------------------------------------------------------------------------------
def test_signal_rules():
    np.testing.assert_array_equal(B.level_position([5, -5, 1, np.nan, -2.5], 2.0),
                                  [-1, 1, 0, 0, 1])
    # same-model delta: nan (first rebalance), +1, -0.1, -1.9 | nan, +0.1 -> flat, short, flat, long | flat, flat
    delta = np.array([np.nan, 1.0, -0.1, -1.9, np.nan, 0.1])
    np.testing.assert_array_equal(B.change_position(delta, 0.5), [0, -1, 0, 1, 0, 0])
    np.testing.assert_array_equal(B.vs_usda_position([5, 5, -1], [np.nan, 1, 4], 2.0),
                                  [0, -1, 1])


def test_metrics_drawdown_and_sharpe():
    daily = pd.DataFrame({"date": pd.bdate_range("2020-06-01", periods=5),
                          "net": [100.0, -300.0, 50.0, 400.0, -100.0]})
    assert B.max_drawdown(daily) == pytest.approx(300.0)
    weekly = pd.DataFrame({"net": [10.0, -5.0, 20.0], "notional": 20000.0})
    r = weekly["net"] / 20000
    assert B.sharpe(weekly) == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(52))


# --- no look-ahead -----------------------------------------------------------------------------
def test_position_at_d_depends_only_on_rows_up_to_d():
    raw, tg = synth_features(seasons=range(2009, 2017))
    feat = M.prepare_features(raw)
    fc = pd.DataFrame({"year": list(range(2009, 2017)) * 2,
                       "period": ["AUG"] * 8 + ["SEP"] * 8,
                       "yield_bu_ac": 170 + np.arange(16, dtype=float),
                       "available_date": [pd.Timestamp(y, 8, 12) for y in range(2009, 2017)]
                       + [pd.Timestamp(y, 9, 12) for y in range(2009, 2017)]})
    td = {y: pd.DatetimeIndex(feat.loc[feat["season"] == y, "date"]) for y in (2014, 2015, 2016)}
    sched = B.build_schedule(td)
    base = B.compute_positions(feat, tg, fc, sched)

    for d in [pd.Timestamp("2015-08-18"), pd.Timestamp("2014-06-10")]:
        assert d in set(sched["date"])
        rng = np.random.default_rng(int(d.day))
        pert = feat.copy()
        later = pert["date"] > d
        cols = [c for c in pert.columns if c not in ("date", "season", "campaign", "doy", "contract")]
        pert.loc[later, cols] = pert.loc[later, cols] * -5 + rng.normal(0, 3, (later.sum(), len(cols)))
        tg_p = tg.copy()
        tg_p.loc[tg_p.index >= d.year, "surprise"] = rng.normal(0, 50, (tg_p.index >= d.year).sum())
        fc_p = fc.copy()
        fc_p.loc[fc_p["available_date"] > d, "yield_bu_ac"] += 40
        new = B.compute_positions(M.prepare_features(pert), tg_p, fc_p, sched)
        upto = sched["date"] <= d
        strat = B.strategy_names(base)
        pd.testing.assert_frame_equal(base.loc[upto, strat], new.loc[upto, strat])
        assert (base.loc[~upto, strat] != new.loc[~upto, strat]).any().any()  # perturbation bites


def test_random_sign_and_bootstrap_shapes():
    rng = np.random.default_rng(0)
    rows = []
    for y in (2014, 2015, 2016):
        for k in range(5):
            rows.append({"season": y, "pos": -1.0, "d_close": rng.normal(-2, 5),
                         "notional": 22000.0})
    w = pd.DataFrame(rows)
    w["gross"] = w["pos"] * w["d_close"] * 50
    w["cost"] = 0.0
    w.loc[w.groupby("season").head(1).index, "cost"] = 30.0
    w["net"] = w["gross"] - w["cost"]
    rt = B.random_sign_tests(w, n=500, seed=0)
    assert 0 < rt["p_random_full"] <= 1 and 0 < rt["p_random_own"] <= 1
    bs = B.block_bootstrap(w, n_boot=500, seed=0, other=w)
    assert bs["pnl_lo"] <= bs["pnl_hi"] and bs["sharpe_lo"] <= bs["sharpe_hi"]
    assert bs["diff"] == 0 and bs["diff_lo"] == 0 and bs["diff_hi"] == 0


# --- change rule: same-model delta, no churn ---------------------------------------------------
def _flat_features(season=2016, seasons=range(2009, 2017)):
    """synth_features, but the rows of ``season`` from Jun 1 are IDENTICAL (features at every
    rebalance equal). Earlier seasons keep their within-season variation, so the ridge
    refit at each rebalance (training rows at that date's day of season) differs."""
    raw, tg = synth_features(seasons=seasons)
    feat = raw.copy()
    cols = [c for c in feat.columns if c not in ("date", "campaign", "contract", "close")]
    m = (feat["campaign"] == season) & (feat["date"] >= f"{season}-06-01")
    feat.loc[m, cols] = feat.loc[m, cols].iloc[0].to_numpy()
    return feat, tg


def test_pure_model_change_gives_zero_delta():
    """Refits differ between rebalances (training rows are taken at each date's day of
    season, so the ridge changes), but with identical features at both dates the
    same-model delta is exactly 0 and the change rule trades nothing."""
    raw, tg = _flat_features()
    feat = M.prepare_features(raw)
    dates = pd.DataFrame({"season": 2016,
                          "date": pd.to_datetime(["2016-06-14", "2016-07-12", "2016-08-16"])})
    p = M.rebalance_predictions(feat, tg, dates, sets=["W", "W+C"])
    for s in ("W", "W+C"):
        g = p[p["set"] == s].sort_values("date")
        assert np.isnan(g["delta"].iloc[0])  # first rebalance of the season
        np.testing.assert_allclose(g["delta"].iloc[1:], 0.0, atol=1e-12)
        # the model (alpha / training rows) did change, so the OLD rule (diff of preds
        # stored at consecutive rebalances) would see churn
        assert g["pred"].diff().abs().iloc[1:].min() > 1e-6
    sig, _ = B.compute_signals(feat, tg, pd.DataFrame(columns=["year", "available_date", "yield_bu_ac"]),
                               dates, sets=["W"])
    assert (B.change_position(sig["delta_W"], 0.0) == 0).all()


def test_delta_uses_same_model_and_handles_missing_columns():
    raw, tg = synth_features(seasons=range(2009, 2017))
    feat = M.prepare_features(raw)
    d0, d1 = pd.Timestamp("2016-07-05"), pd.Timestamp("2016-08-02")
    dates = pd.DataFrame({"season": 2016, "date": [d0, d1]})
    p = M.rebalance_predictions(feat, tg, dates, sets=["W"]).set_index("date")
    by = feat.set_index("date")
    # reference: fit at d1 and score both rows with that one model
    with_t = M._with_targets(tg)
    ax = M.AsofIndex(feat)
    tX = ax.rows(M.training_pool("wf", 2016, with_t), int(by.loc[d1, "doy"]))
    ty = tg.loc[tX.index, "surprise"]
    cols = M.FEATURE_SETS["W"]
    model = M.fit_ridge(tX[cols], ty)
    ref = (model.predict(pd.DataFrame([by.loc[d1, cols].astype(float).to_numpy()], columns=cols))[0]
           - model.predict(pd.DataFrame([by.loc[d0, cols].astype(float).to_numpy()], columns=cols))[0])
    assert p.loc[d1, "delta"] == pytest.approx(ref)
    assert p.loc[d1, "delta"] == pytest.approx(p.loc[d1, "pred"] - p.loc[d1, "pred_prev"])
    # previous row lacks a column the model uses -> NaN delta -> flat
    feat2 = feat.copy()
    feat2.loc[feat2["date"] == d0, "edd30_cum"] = np.nan
    p2 = M.rebalance_predictions(feat2, tg, dates, sets=["W"]).set_index("date")
    assert np.isnan(p2.loc[d1, "delta"]) and p2.loc[d1, "used"] == p.loc[d1, "used"]
    assert B.change_position(p2["delta"].to_numpy(), 0.5).tolist() == [0, 0]
    # a new season restarts: its first rebalance has no previous row
    d2 = pd.DataFrame({"season": [2015, 2016], "date": [pd.Timestamp("2015-08-04"), d0]})
    p3 = M.rebalance_predictions(feat, tg, d2, sets=["W"])
    assert p3["delta"].isna().all()


# --- H (HLS) trade sets ---------------------------------------------------------------------------
def _signals_inputs(**synth_kw):
    raw, tg = synth_features(seasons=range(2009, 2017), **synth_kw)
    feat = M.prepare_features(raw)
    fc = pd.DataFrame({"year": [2009], "period": ["AUG"], "yield_bu_ac": [170.0],
                       "available_date": [pd.Timestamp(2009, 8, 12)]})
    td = {y: pd.DatetimeIndex(feat.loc[feat["season"] == y, "date"]) for y in (2015, 2016)}
    return feat, tg, fc, B.build_schedule(td)


def test_trade_sets_include_h_and_skip_it_without_hls_columns():
    assert {"H", "W+C+H"} <= set(B.TRADE_SETS)
    feat, tg, fc, sched = _signals_inputs()
    sig, long = B.compute_signals(feat, tg, fc, sched)
    assert B.signal_sets(sig) == ["W", "C", "W+C", "W+C+S", "S"]  # same as before H existed
    assert {"sdrop_W+C+S", "hdrop_W+C+S"} <= set(sig.columns)
    assert not [c for c in sig.columns if c.endswith("+H") or c.endswith("_H")]
    pos = B.positions_table(sig)
    assert not [c for c in B.strategy_names(pos) if c.split("|")[-1].split("+")[-1] == "H"]
    assert set(long["set"]) == {"W", "C", "W+C", "W+C+S", "S"}


def test_h_signals_and_hdrop_flags_with_hls_columns():
    feat, tg, fc, sched = _signals_inputs(hls_from=2009)
    sig, _ = B.compute_signals(feat, tg, fc, sched)
    assert {"pred_H", "pred_W+C+H", "hdrop_H", "hdrop_W+C+H", "sdrop_H"} <= set(sig.columns)
    assert sig["pred_W+C+H"].notna().any() and not sig["hdrop_W+C+H"].astype(bool).any()
    assert not sig["sdrop_W+C+H"].astype(bool).any()
    pos = B.positions_table(sig)
    assert {"level|H", "change|W+C+H", "vs_usda|H"} <= set(B.strategy_names(pos))
    # non-H signals are the same with or without the hls columns
    feat0, tg0, fc0, sched0 = _signals_inputs()
    sig0, _ = B.compute_signals(feat0, tg0, fc0, sched0)
    pd.testing.assert_frame_equal(sig0, sig[sig0.columns])


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # 1-week toy seasons: Sharpe undefined
def test_sat_shared_table_uses_only_seasons_where_both_predict():
    seasons = [2018, 2019, 2020, 2021]
    sig = pd.DataFrame({"season": np.repeat(seasons, 2),
                        "date": pd.to_datetime([f"{y}-06-{d:02d}" for y in seasons for d in (2, 9)])})
    nan = np.nan
    sig["pred_W+C"] = 1.0
    sig["pred_W+C+S"], sig["sdrop_W+C+S"] = 1.0, False
    sig["pred_S"], sig["sdrop_S"] = 1.0, False
    # H: no H history in 2018 (dropped -> = W+C), NaN in 2019, valid 2020-21
    sig["pred_W+C+H"] = 1.0
    sig["hdrop_W+C+H"] = sig["season"] <= 2018
    sig["pred_H"] = np.where(sig["season"] <= 2019, nan, 1.0)
    sig["hdrop_H"] = sig["season"] <= 2019
    # S dropped on every 2021 rebalance -> 2021 not shared either
    sig.loc[sig["season"] == 2021, "sdrop_W+C+S"] = True
    assert B.valid_prediction_seasons(sig, "W+C+H") == {2019, 2020, 2021}
    assert B.shared_sat_seasons(sig) == [2020]
    pnl = {"W+C": 10.0, "W+C+S": 30.0, "S": -5.0, "W+C+H": 50.0, "H": 0.0}
    res = {}
    for s, v in pnl.items():
        w = pd.DataFrame({"season": seasons, "net": [v + 1000 * (y != 2020) for y in seasons],
                          "notional": 20_000.0})
        res[f"level|{s}"] = (None, w)
    t = B.sat_shared_table(res, sig, n_boot=200, signals_list=["level"]).iloc[0]
    assert t["n_seasons"] == 1 and t["seasons"] == "2020"
    assert t["pnl_W+C+S"] == 30 and t["pnl_W+C+H"] == 50 and t["pnl_W+C"] == 10
    assert t["diff_WCH_minus_WCS"] == 20 and t["diff_W+C+S_vs_W+C"] == 20
    assert t["seasons_H_better"] == 1 and t["seasons_S_better"] == 0

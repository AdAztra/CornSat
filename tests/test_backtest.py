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
    pred = pd.Series([1.0, 2.0, 1.9, 0.0, 3.0, 3.1])
    season = pd.Series([2020, 2020, 2020, 2020, 2021, 2021])
    # delta: nan, +1, -0.1, -1.9 | nan, +0.1 -> short, flat, long | flat, flat
    np.testing.assert_array_equal(B.change_position(pred, season, 0.5), [0, -1, 0, 1, 0, 0])
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

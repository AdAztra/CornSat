"""Phase 6 Stage A tests: trend uses only prior years, walk-forward trains only on prior
seasons, standardisation is fit on training data only, missing-feature fallback."""

import numpy as np
import pandas as pd
import pytest

from cornsat import model as M

ALL_FEATURES = sorted({f for fs in M.FEATURE_GROUPS.values() for f in fs})


def synth_final(first=1990, last=2026, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    yrs = np.arange(first, last + 1)
    return pd.Series(120 + 1.9 * (yrs - first) + rng.normal(0, 8, len(yrs)), index=yrs,
                     name="final")


def synth_features(seasons=range(2009, 2021), seed=3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Business-day feature table (Apr-Oct) whose features carry a per-season latent u,
    plus targets with surprise = 5 u + noise."""
    rng = np.random.default_rng(seed)
    parts, tg = [], []
    for y in seasons:
        u = rng.normal()
        d = pd.bdate_range(f"{y}-04-01", f"{y}-10-31")
        ramp = np.linspace(0, 1, len(d))
        f = pd.DataFrame({"date": d, "campaign": y, "contract": f"ZCZ{y % 100:02d}",
                          "close": 450 + rng.normal(0, 5, len(d)).cumsum()})
        for i, c in enumerate(ALL_FEATURES):
            f[c] = u * ramp * (1 + 0.1 * i) + rng.normal(0, 0.5, len(d))
        parts.append(f)
        tg.append({"season": y, "final": np.nan, "trend": 170.0, "surprise": 5 * u + rng.normal()})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(tg).set_index("season")


# --- trend ---------------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["ols", "huber"])
def test_trend_uses_only_years_before_season(method):
    final = synth_final()
    for season in (2009, 2015, 2026):
        val, years = M.trend_for_season(final, season, method=method)
        assert years and max(years) < season and min(years) >= M.TREND_START
        bumped = final.copy()
        bumped[bumped.index >= season] += 1000.0  # perturb the season and later
        val2, _ = M.trend_for_season(bumped, season, method=method)
        assert val == pytest.approx(val2, abs=1e-9)


def test_season_targets_trend_years():
    final = synth_final()
    tg = M.season_targets(final, range(2009, 2027))
    assert (tg["trend_last"] == tg.index - 1).all()
    assert (tg["trend_first"] == M.TREND_START).all()
    np.testing.assert_allclose(tg["surprise"], tg["final"] - tg["trend"])


def test_usda_surprise_uses_only_released_forecasts():
    tg = pd.DataFrame({"trend": [170.0]}, index=pd.Index([2020], name="season"))
    fc = pd.DataFrame({"year": [2020, 2020], "period": ["AUG", "SEP"],
                       "yield_bu_ac": [175.0, 180.0],
                       "available_date": pd.to_datetime(["2020-08-13", "2020-09-14"])})
    assert np.isnan(M.usda_surprise_at(fc, tg, 2020, "2020-08-12"))
    assert M.usda_surprise_at(fc, tg, 2020, "2020-08-13") == 5.0
    assert M.usda_surprise_at(fc, tg, 2020, "2020-09-13") == 5.0
    assert M.usda_surprise_at(fc, tg, 2020, "2020-09-14") == 10.0


# --- walk-forward training sets --------------------------------------------------------------
def test_walk_forward_snapshot_training_seasons_strictly_before_target():
    raw, tg = synth_features()
    feat = M.prepare_features(raw)
    fc = pd.DataFrame({"year": pd.Series(dtype=int), "period": pd.Series(dtype=str),
                       "yield_bu_ac": pd.Series(dtype=float),
                       "available_date": pd.Series(dtype="datetime64[ns]")})
    preds = M.snapshot_predictions(feat, tg, fc, snapshots={"Aug01": (8, 1)},
                                   sets=["W", "W+C+S"])
    model_rows = preds[preds["set"].isin(["W", "W+C+S"])]
    wf = model_rows[model_rows["scheme"] == "wf"]
    assert wf["season"].min() == 2009 + M.MIN_TRAIN
    for _, r in wf.iterrows():
        assert len(r["train_seasons"]) >= M.MIN_TRAIN
        assert max(r["train_seasons"]) < r["season"], r
        assert r["train_seasons"] == list(range(2009, r["season"]))
    for _, r in model_rows[model_rows["scheme"] == "loso"].iterrows():
        assert r["season"] not in r["train_seasons"]
    mean_rows = preds[(preds["set"] == "mean") & (preds["scheme"] == "wf")]
    assert all(max(r) < s for r, s in zip(mean_rows["train_seasons"], mean_rows["season"]))


def test_walk_forward_rebalance_training_seasons_strictly_before_target():
    raw, tg = synth_features()
    feat = M.prepare_features(raw)
    dates = pd.DataFrame({"season": [2014, 2016, 2020],
                          "date": pd.to_datetime(["2014-06-03", "2016-08-02", "2020-09-15"])})
    p = M.rebalance_predictions(feat, tg, dates, sets=["W", "S"])
    for _, r in p.iterrows():
        assert r["train_seasons"] and max(r["train_seasons"]) < r["season"]


def test_asof_rows_use_same_day_of_season():
    raw, _ = synth_features(seasons=range(2009, 2012))
    feat = M.prepare_features(raw)
    rows = M.AsofIndex(feat).rows([2009, 2010, 2011], int(M.cal_doy(["2011-07-04"])[0]))
    # Jul 4 2009 is a Saturday -> Fri Jul 3; 2010 Sunday -> Fri Jul 2; 2011 Monday -> itself
    assert list(rows["date"]) == list(pd.to_datetime(["2009-07-03", "2010-07-02", "2011-07-04"]))


# --- standardisation ---------------------------------------------------------------------------
def test_standardisation_fit_on_training_rows_only():
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(9, 3)) * [1, 10, 100] + [0, 5, -50], columns=list("abc"),
                     index=range(2009, 2018))
    y = pd.Series(X["a"] * 2 - X["b"] * 0.1 + rng.normal(0, 0.1, 9), index=X.index)
    model = M.fit_ridge(X, y, alphas=(5.0,))
    np.testing.assert_allclose(model["scaler"].mean_, X.mean().to_numpy())
    np.testing.assert_allclose(model["scaler"].scale_, X.std(ddof=0).to_numpy())

    # predict_one == closed-form ridge on training-only z-scores (no intercept)
    x = pd.Series({"a": 50.0, "b": -300.0, "c": 1e4})  # extreme target row
    out = M.predict_one(X, y, x, ["a", "b", "c"], alphas=(5.0,), min_train=5)
    mu, sd = X.mean().to_numpy(), X.std(ddof=0).to_numpy()
    Z = (X.to_numpy() - mu) / sd
    w = np.linalg.solve(Z.T @ Z + 5.0 * np.eye(3), Z.T @ y.to_numpy())
    assert out["pred"] == pytest.approx(float(((x.to_numpy() - mu) / sd) @ w), rel=1e-9)

    # if the target row had leaked into the scaler, the prediction would differ
    Xl = pd.concat([X, x.to_frame().T])
    mu_l, sd_l = Xl.mean().to_numpy(), Xl.std(ddof=0).to_numpy()
    Zl = (X.to_numpy() - mu_l) / sd_l
    wl = np.linalg.solve(Zl.T @ Zl + 5.0 * np.eye(3), Zl.T @ y.to_numpy())
    assert abs(float(((x.to_numpy() - mu_l) / sd_l) @ wl) - out["pred"]) > 1e-3


def test_no_signal_shrinks_to_trend():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(12, 2)), columns=["a", "b"], index=range(2009, 2021))
    y = pd.Series(rng.normal(size=12), index=X.index)
    out = M.predict_one(X, y, pd.Series({"a": 0.5, "b": -0.5}), ["a", "b"], alphas=(1e9,))
    assert abs(out["pred"]) < 1e-6  # no intercept: heavy shrinkage -> 0 = the trend


# --- fallback ----------------------------------------------------------------------------------
def test_missing_satellite_falls_back_to_model_without_s():
    raw, tg = synth_features()
    feat = M.prepare_features(raw)
    snap = M.snapshot_rows(feat, 8, 15)
    pool = list(range(2009, 2020))
    x = snap.loc[2020].copy()
    x[M.FEATURE_GROUPS["S"]] = np.nan  # e.g. 2025 MODIS outage
    full = M.predict_one(snap.loc[pool], tg.loc[pool, "surprise"], x, M.FEATURE_SETS["W+C+S"])
    base = M.predict_one(snap.loc[pool], tg.loc[pool, "surprise"], x, M.FEATURE_SETS["W+C"])
    assert full["fallback"] and full["s_dropped"]
    assert full["used"] == base["used"] == ",".join(M.FEATURE_SETS["W+C"])
    assert full["pred"] == pytest.approx(base["pred"])
    s_only = M.predict_one(snap.loc[pool], tg.loc[pool, "surprise"], x, M.FEATURE_SETS["S"])
    assert np.isnan(s_only["pred"]) and s_only["s_dropped"]


def test_feature_missing_in_training_is_dropped_not_seasons():
    raw, tg = synth_features()
    feat = M.prepare_features(raw)
    snap = M.snapshot_rows(feat, 6, 15)
    pool = list(range(2009, 2016))
    tX = snap.loc[pool].copy()
    tX.loc[[2009, 2011, 2013], "ge_chg_2w"] = np.nan  # only 4 complete seasons with it
    out = M.predict_one(tX, tg.loc[pool, "surprise"], snap.loc[2016], M.FEATURE_SETS["C"])
    assert out["used"] == "ge" and out["n_train"] == 7 and out["fallback"]

"""Phase 6 Stage A tests: trend uses only prior years, walk-forward trains only on prior
seasons, standardisation is fit on training data only, missing-feature fallback."""

import numpy as np
import pandas as pd
import pytest

from cornsat import model as M

ALL_FEATURES = sorted({f for g, fs in M.FEATURE_GROUPS.items() if g != "H" for f in fs})
HLS_FEATURES = M.FEATURE_GROUPS["H"]


def synth_final(first=1990, last=2026, seed=0) -> pd.Series:
    rng = np.random.default_rng(seed)
    yrs = np.arange(first, last + 1)
    return pd.Series(120 + 1.9 * (yrs - first) + rng.normal(0, 8, len(yrs)), index=yrs,
                     name="final")


def synth_features(seasons=range(2009, 2021), seed=3, hls_from: int | None = None,
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Business-day feature table (Apr-Oct) whose features carry a per-season latent u,
    plus targets with surprise = 5 u + noise. No hls_ columns unless ``hls_from`` is given:
    then they exist, carrying the same latent u, but are NaN for seasons < ``hls_from``
    (drawn from a separate rng, so the other columns are the same either way)."""
    rng = np.random.default_rng(seed)
    rng_h = np.random.default_rng(seed + 1000)
    parts, tg = [], []
    for y in seasons:
        u = rng.normal()
        d = pd.bdate_range(f"{y}-04-01", f"{y}-10-31")
        ramp = np.linspace(0, 1, len(d))
        f = pd.DataFrame({"date": d, "campaign": y, "contract": f"ZCZ{y % 100:02d}",
                          "close": 450 + rng.normal(0, 5, len(d)).cumsum()})
        for i, c in enumerate(ALL_FEATURES):
            f[c] = u * ramp * (1 + 0.1 * i) + rng.normal(0, 0.5, len(d))
        if hls_from is not None:
            for i, c in enumerate(HLS_FEATURES):
                f[c] = (u * ramp * (1 + 0.1 * i) + rng_h.normal(0, 0.5, len(d))
                        if y >= hls_from else np.nan)
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


# --- H (HLS) group -------------------------------------------------------------------------------
def test_h_group_and_sets_registered():
    assert M.FEATURE_GROUPS["H"] == ["hls_ndvi_z", "hls_evi_anom_cum", "hls_ndvi_trend_anom"]
    assert M.SET_NAMES[:7] == ["W", "C", "S", "W+C", "W+S", "C+S", "W+C+S"]  # order unchanged
    assert M.SET_NAMES[7:] == ["H", "W+H", "C+H", "W+C+H", "W+C+S+H"]
    assert M.SAT_GROUPS == ("S", "H") and M.S_FEATURES == set(M.FEATURE_GROUPS["S"])
    assert M.FEATURE_SETS["W+C+S+H"] == [*M.FEATURE_SETS["W+C+S"], *M.FEATURE_GROUPS["H"]]


def _stage_a(feat_raw, tg):
    feat = M.prepare_features(feat_raw)
    fc = pd.DataFrame(columns=["year", "period", "yield_bu_ac", "available_date"])
    fc["available_date"] = pd.to_datetime(fc["available_date"])
    return feat, M.snapshot_predictions(feat, tg, fc, snapshots={"Aug15": (8, 15)})


def test_without_hls_columns_h_sets_skipped_and_s_outputs_unchanged(caplog):
    raw, tg = synth_features()
    assert not any(c.startswith("hls_") for c in raw.columns)
    M._WARNED_MISSING.clear()
    with caplog.at_level("WARNING", logger="cornsat.model"):
        feat, preds = _stage_a(raw, tg)
        # a second consumer in the same process must not log again
        M.rebalance_predictions(feat, tg, pd.DataFrame(
            {"season": [2019], "date": [feat[feat["season"] == 2019]["date"].iloc[60]]}))
    msgs = [r for r in caplog.records if "skipping feature sets" in r.getMessage()]
    assert len(msgs) == 1 and "hls_ndvi_z" in msgs[0].getMessage()
    sets = set(preds["set"])
    assert not any("H" in s.split("+") for s in sets)
    assert {"W", "C", "S", "W+C", "W+S", "C+S", "W+C+S"} <= sets
    # the S-set predictions equal an independent per-set predict_one on the same data
    snap = M.snapshot_rows(feat, 8, 15)
    with_t = [int(y) for y in tg.index]
    for s in ("S", "W+C+S"):
        for Y in (2015, 2019):
            pool = [y for y in with_t if y < Y]
            ref = M.predict_one(snap.loc[pool], tg.loc[pool, "surprise"], snap.loc[Y],
                                M.FEATURE_SETS[s])
            got = preds[(preds["scheme"] == "wf") & (preds["set"] == s) & (preds["season"] == Y)]
            assert got["pred"].iloc[0] == pytest.approx(ref["pred"])
            assert not got["h_dropped"].iloc[0]
    # adding hls_ columns does not change any non-H output
    raw_h, _ = synth_features(hls_from=2009)
    _, preds_h = _stage_a(raw_h, tg)
    a = preds[preds["set"].isin(sets)].drop(columns=["train_seasons"]).reset_index(drop=True)
    b = preds_h[preds_h["set"].isin(sets)].drop(columns=["train_seasons"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_h_sets_run_with_hls_columns_and_h_dropped_flag():
    raw, tg = synth_features(hls_from=2009)
    feat, preds = _stage_a(raw, tg)
    assert {"H", "W+H", "C+H", "W+C+H", "W+C+S+H"} <= set(preds["set"])
    snap = M.snapshot_rows(feat, 8, 15)
    pool = list(range(2009, 2019))
    x = snap.loc[2019].copy()
    x[M.FEATURE_GROUPS["H"]] = np.nan  # HLS outage in the target row
    full = M.predict_one(snap.loc[pool], tg.loc[pool, "surprise"], x, M.FEATURE_SETS["W+C+H"])
    base = M.predict_one(snap.loc[pool], tg.loc[pool, "surprise"], x, M.FEATURE_SETS["W+C"])
    assert full["h_dropped"] and not full["s_dropped"] and full["fallback"]
    assert full["pred"] == pytest.approx(base["pred"])
    assert not base["h_dropped"] and not base["s_dropped"]
    # dropping only S leaves h_dropped False
    x2 = snap.loc[2019].copy()
    x2[M.FEATURE_GROUPS["S"]] = np.nan
    both = M.predict_one(snap.loc[pool], tg.loc[pool, "surprise"], x2, M.FEATURE_SETS["W+C+S+H"])
    assert both["s_dropped"] and not both["h_dropped"]
    h_only = M.predict_one(snap.loc[pool], tg.loc[pool, "surprise"], x, M.FEATURE_SETS["H"])
    assert np.isnan(h_only["pred"]) and h_only["h_dropped"] and not h_only["s_dropped"]


def test_h_predictions_start_late_and_paired_uses_season_intersection():
    raw, tg = synth_features(seasons=range(2009, 2027), hls_from=2016)
    tg.loc[2026, "surprise"] = np.nan  # no target yet
    feat, preds = _stage_a(raw, tg)
    wf = preds[(preds["scheme"] == "wf")]
    h_seasons = sorted(wf[(wf["set"] == "H") & wf["pred"].notna()]["season"])
    w_seasons = sorted(wf[(wf["set"] == "W") & wf["pred"].notna()]["season"])
    assert h_seasons[0] == 2016 + M.MIN_TRAIN and w_seasons[0] == 2009 + M.MIN_TRAIN
    pc = M.paired_comparison(preds, scheme="wf")
    pairs = set(zip(pc["base"], pc["with_s"]))
    assert {("W+C", "W+C+H"), ("W+C+S", "W+C+H"), ("S", "H"), ("W", "W+H"),
            ("C", "C+H"), ("W+C", "W+C+S")} <= pairs
    allrow = pc[(pc["base"] == "W") & (pc["with_s"] == "W+H") & (pc["snapshot"] == "ALL")].iloc[0]
    # intersection: H seasons that have a target, not all of W's seasons
    assert allrow["n"] == len([s for s in h_seasons if s <= 2025])
    assert allrow["seasons"] == f"{h_seasons[0]}-2025"
    wc = pc[(pc["base"] == "W") & (pc["with_s"] == "W+S") & (pc["snapshot"] == "ALL")].iloc[0]
    assert wc["n"] > allrow["n"] and wc["seasons"].startswith(str(w_seasons[0]))


def test_paired_comparison_skips_pairs_with_missing_sets():
    raw, tg = synth_features()
    _, preds = _stage_a(raw, tg)  # no hls columns -> no H sets
    pc = M.paired_comparison(preds, scheme="wf")
    assert not pc["with_s"].str.endswith("H").any()
    assert set(zip(pc["base"], pc["with_s"])) == {("W+C", "W+C+S"), ("W", "W+S"), ("C", "C+S")}
    assert "seasons" in pc.columns
    assert M.paired_comparison(preds.iloc[0:0], scheme="wf").empty

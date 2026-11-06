"""Phase 8f state-panel tests (synthetic): state trends use only prior years, LOSO drops
every row of the held-out season, walk-forward trains on earlier seasons only, scaler and
alpha are fit on training rows only (alpha by grouped leave-one-season-out), comparisons
use identical rows, event timing uses prior-season sigma only."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cornsat import panel as P

STATES = ["IA", "IL", "NE", "MN", "IN"]
SEASONS = list(range(2009, 2023))
SNAPS = ["Jul15", "Aug15"]


def synth_data(seasons=SEASONS, seed=0, groups=("W", "C", "S")) -> pd.DataFrame:
    """Rows (state, season, snapshot) whose features carry a season shock u and a
    state shock v; surprise = 6 u + 3 v + noise."""
    rng = np.random.default_rng(seed)
    feats = [f for g in groups for f in P.GROUPS[g]]
    rows = []
    for y in seasons:
        u = rng.normal()
        for s in STATES:
            v = rng.normal()
            sur = 6 * u + 3 * v + rng.normal()
            for k, snap in enumerate(SNAPS):
                r = {"state": s, "season": y, "snapshot": snap, "surprise": sur,
                     "usda": sur + rng.normal(0, 2) if snap == "Aug15" else np.nan}
                for i, f in enumerate(feats):
                    r[f] = (u + 0.5 * v) * (1 + 0.1 * i) * (k + 1) + rng.normal(0, 0.7)
                rows.append(r)
    return pd.DataFrame(rows)


def synth_finals(first=1990, last=2025, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    yrs = np.arange(first, last + 1)
    return pd.DataFrame({s: 120 + (1.5 + 0.1 * i) * (yrs - first) + rng.normal(0, 9, len(yrs))
                         for i, s in enumerate(STATES)}, index=pd.Index(yrs, name="year"))


# --- targets ---------------------------------------------------------------------------------
def test_state_trend_uses_only_years_before_season():
    fin = synth_finals()
    tg = P.state_targets(fin, range(2009, 2026))
    assert (tg["trend_last"] == tg["season"] - 1).all()
    np.testing.assert_allclose(tg["surprise"], tg["final"] - tg["trend"])
    for Y in (2009, 2016, 2025):
        bumped = fin.copy()
        bumped.loc[bumped.index >= Y] += 1000.0
        a = tg[tg["season"] == Y].set_index("state")["trend"]
        b = P.state_targets(bumped, [Y]).set_index("state")["trend"]
        pd.testing.assert_series_equal(a, b, check_exact=False, atol=1e-9)
    # each state is detrended on its own history
    one = P.state_targets(fin[["IA"]], [2020])["trend"].iloc[0]
    assert one == pytest.approx(tg[(tg["state"] == "IA") & (tg["season"] == 2020)]["trend"].iloc[0])


def test_state_finals_release_aware_and_sample_wins():
    u = pd.DataFrame({"geo": ["IA", "IA", "IA", "IL"], "year": [2009, 2010, 2010, 2010],
                      "period": ["FINAL", "FINAL", "AUG", "FINAL"],
                      "yield_bu_ac": [182.0, 165.0, 170.0, 157.0],
                      "available_date": pd.to_datetime(["2010-01-12", "2011-01-12",
                                                        "2010-08-12", "2011-01-12"])})
    hist = pd.DataFrame({"state": ["IA", "IA"], "year": [2008, 2009], "final": [171.0, 999.0]})
    f = P.state_final_yields(u, hist, asof="2010-12-31", states=["IA", "IL"])
    assert f.loc[2009, "IA"] == 182.0  # sample beats history on overlap
    assert f.loc[2008, "IA"] == 171.0
    assert 2010 not in f.index  # finals of 2010 not yet released on 2010-12-31


# --- validation schemes ----------------------------------------------------------------------
FEATS = P.set_features("W+C+S")


def test_loso_removes_every_row_of_the_held_out_season(monkeypatch):
    d = synth_data()
    seen = []
    orig = P.select_alpha

    def spy(X, y, season, *a, **k):
        seen.append(set(np.asarray(season).tolist()))
        return orig(X, y, season, *a, **k)

    monkeypatch.setattr(P, "select_alpha", spy)
    p = P.cv_predictions(d, FEATS, "loso")
    n_snap = d.groupby("snapshot").size()
    for (snap, Y), g in p.groupby(["snapshot", "season"]):
        assert (g["n_train"] == n_snap[snap] - (d["season"] == Y).sum() / len(SNAPS)).all()
    assert all(len(s) == len(SEASONS) - 1 for s in seen)
    # targets of the held-out season (all states) never reach its own predictions
    Y = 2015
    d2 = d.copy()
    d2.loc[d2["season"] == Y, "surprise"] += 500.0
    monkeypatch.setattr(P, "select_alpha", orig)
    a = P.cv_predictions(d, FEATS, "loso").query("season == @Y")["pred"].to_numpy()
    b = P.cv_predictions(d2, FEATS, "loso").query("season == @Y")["pred"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_walk_forward_trains_only_on_earlier_seasons():
    d = synth_data()
    p = P.cv_predictions(d, FEATS, "wf")
    assert (p["train_last"] < p["season"]).all()
    assert p["season"].min() == SEASONS[P.MIN_TRAIN_SEASONS]
    assert (p["n_train_seasons"] == p["season"] - SEASONS[0]).all()
    Y = 2016
    d2 = d.copy()
    later = d2["season"] > Y
    d2.loc[later, FEATS] *= -50.0
    d2.loc[d2["season"] >= Y, "surprise"] += 300.0
    a = P.cv_predictions(d, FEATS, "wf").query("season == @Y")["pred"].to_numpy()
    b = P.cv_predictions(d2, FEATS, "wf").query("season == @Y")["pred"].to_numpy()
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize("demean", [False, True])
def test_scaler_and_alpha_fit_on_training_rows_only(demean):
    d = synth_data()
    Y, snap = 2014, "Aug15"
    g = d[d["snapshot"] == snap]
    tr, te = g[g["season"] != Y], g[g["season"] == Y]
    a = P.select_alpha(tr[FEATS].to_numpy(), tr["surprise"].to_numpy(), tr["season"].to_numpy(),
                       tr["state"].to_numpy(), demean=demean)
    m = P.PooledRidge(a, demean).fit(tr[FEATS].to_numpy(), tr["surprise"].to_numpy(),
                                     tr["state"].to_numpy())
    np.testing.assert_allclose(m.mean_, (tr[FEATS] - (tr.groupby("state")[FEATS].transform("mean")
                                                      if demean else 0)).mean().to_numpy(),
                               atol=1e-12)
    # held-out rows wildly rescaled: same alpha, same fitted model applied to the new x
    d2 = d.copy()
    d2.loc[(d2["season"] == Y), FEATS] *= 40.0
    for data, x in ((d, te), (d2, d2[(d2["season"] == Y) & (d2["snapshot"] == snap)])):
        p = P.cv_predictions(data, FEATS, "loso", demean=demean).query(
            "season == @Y and snapshot == @snap")
        assert (p["alpha"] == a).all()
        np.testing.assert_allclose(p["pred"].to_numpy(),
                                   m.predict(x[FEATS].to_numpy(), x["state"].to_numpy()))


def test_pooled_ridge_matches_sklearn():
    d = synth_data()
    X, y = d[FEATS].to_numpy(), d["surprise"].to_numpy()
    ref = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=3.0, fit_intercept=False))])
    ref.fit(X, y)
    np.testing.assert_allclose(P.PooledRidge(3.0).fit(X, y).predict(X), ref.predict(X), atol=1e-8)


def test_alpha_selected_by_grouped_leave_one_season_out(monkeypatch):
    d = synth_data(seasons=range(2009, 2015))
    g = d[d["snapshot"] == "Aug15"]
    X, y, season = g[FEATS].to_numpy(), g["surprise"].to_numpy(), g["season"].to_numpy()
    held = []
    orig = P.PooledRidge.predict

    def spy(self, Xt, state=None):
        held.append(len(Xt))
        return orig(self, Xt, state)

    monkeypatch.setattr(P.PooledRidge, "predict", spy)
    a = P.select_alpha(X, y, season)
    assert set(held) == {len(STATES)}  # every fold holds out one whole season (5 rows)
    assert len(held) == len(P.ALPHAS) * 6
    monkeypatch.setattr(P.PooledRidge, "predict", orig)
    sse = [sum(((P.PooledRidge(al).fit(X[season != s], y[season != s]).predict(X[season == s])
                 - y[season == s]) ** 2).sum() for s in np.unique(season)) for al in P.ALPHAS]
    assert a == P.ALPHAS[int(np.argmin(sse))]


# --- comparisons -----------------------------------------------------------------------------
def test_common_sample_comparison_uses_identical_rows():
    d = synth_data()
    rng = np.random.default_rng(5)
    d.loc[rng.random(len(d)) < 0.2, "ndvi_z"] = np.nan  # S missing on some rows
    d.loc[rng.random(len(d)) < 0.1, "ge"] = np.nan
    windows = {"2009+": (2009, ("W", "C", "S")), "2014+": (2014, ("W", "C", "S"))}
    pairs = [("W+C", "W+C+S"), ("trend", "W+C+S")]
    preds, _, pair, _, skipped = P.compare(d, windows=windows, pairs=pairs, n_boot=200)
    assert skipped.empty
    for (w, sch), g in preds.groupby(["window", "scheme"]):
        keys = {s: set(map(tuple, gg[["state", "season", "snapshot"]].to_numpy()))
                for s, gg in g.groupby("set")}
        assert keys["W+C"] == keys["W+C+S"] == keys["trend"]
        assert g["season"].min() >= windows[w][0]
        merged = g.merge(d, on=["state", "season", "snapshot"])
        assert merged["ndvi_z"].notna().all() and merged["ge"].notna().all()
    allp = pair[pair["snapshot"] == "ALL"]
    assert (allp.groupby(["window", "scheme"])["n_rows"].nunique() == 1).all()


def test_paired_averages_per_season_first():
    rows = []
    for y, (ea, eb) in {2018: (3.0, 1.0), 2019: (1.0, 2.0), 2020: (2.0, 2.0)}.items():
        for s in STATES[: (2 if y == 2018 else 5)]:
            rows += [{"scheme": "wf", "set": "A", "state": s, "season": y, "snapshot": "Aug15",
                      "pred": ea, "actual": 0.0},
                     {"scheme": "wf", "set": "B", "state": s, "season": y, "snapshot": "Aug15",
                      "pred": eb, "actual": 0.0}]
    r = P.paired(pd.DataFrame(rows), pairs=[("A", "B")], n_boot=200)
    r = r[r["snapshot"] == "ALL"].iloc[0]
    assert r["n_seasons"] == 3 and r["n_rows"] == 12
    assert r["mean_d"] == pytest.approx(((9 - 1) + (1 - 4) + 0) / 3)
    assert r["n_seasons_B_better"] == 1


def test_hls_group_used_once_columns_exist():
    d = synth_data(groups=("W", "C", "S", "H"), seasons=range(2014, 2024))
    assert "H" in P.available_groups(d)
    windows = {"2016+": (2016, ("W", "C", "S", "H"))}
    _, _, pair, _, skipped = P.compare(d, windows=windows, n_boot=100)
    assert {("W+C", "W+C+H"), ("S", "H")} <= set(zip(pair["A"], pair["B"]))
    assert skipped.empty
    d0 = synth_data(seasons=range(2014, 2024))
    _, _, pair0, _, skipped0 = P.compare(d0, windows=windows, n_boot=100)
    assert not pair0["B"].str.contains("H").any() and len(skipped0)


def test_build_dataset_snapshot_and_usda_release_aware():
    dates = pd.date_range("2020-06-01", "2020-09-30", freq="D")
    panel = pd.DataFrame({"geo": "IA", "date": dates, "season": 2020,
                          **{f: np.arange(len(dates), dtype=float) for f in P.set_features("W+C+S")}})
    panel = panel[panel["date"] != "2020-08-15"]  # snapshot falls back to the day before
    tg = pd.DataFrame({"state": ["IA"], "season": [2020], "final": [178.0], "trend": [190.0],
                       "surprise": [-12.0]})
    fc = pd.DataFrame({"state": ["IA", "IA"], "year": [2020, 2020], "yield_bu_ac": [195.0, 177.0],
                       "available_date": pd.to_datetime(["2020-08-13", "2020-09-14"])})
    snaps = {"Aug01": (8, 1), "Aug15": (8, 15), "Sep15": (9, 15)}
    d = P.build_dataset(panel, tg, fc, snapshots=snaps).set_index("snapshot")
    assert d.loc["Aug15", "date"] == pd.Timestamp("2020-08-14")
    assert np.isnan(d.loc["Aug01", "usda"])
    assert d.loc["Aug15", "usda"] == pytest.approx(5.0)
    assert d.loc["Sep15", "usda"] == pytest.approx(-13.0)


def test_long_from_trading():
    t = pd.DataFrame({"date": pd.to_datetime(["2020-07-01", "2020-07-02"]), "campaign": 2020,
                      "ndvi_z": [9.0, 9.0], "ndvi_z_ia": [1.0, 2.0], "ndvi_z_il": [3.0, 4.0]})
    p = P.long_from_trading(t, states=["IA", "IL"])
    assert list(p["geo"]) == ["IA", "IA", "IL", "IL"]
    assert list(p["ndvi_z"]) == [1.0, 2.0, 3.0, 4.0]


# --- event timing ----------------------------------------------------------------------------
def _event_panel(seed=2):
    rng = np.random.default_rng(seed)
    parts = []
    for y in range(2009, 2026):
        d = pd.date_range(f"{y}-06-01", f"{y}-10-31", freq="D")
        v = rng.normal(0, 1, len(d)) * 0.2 + rng.normal()
        if y == 2020:  # normal season, then a collapse from Aug 20
            v = rng.normal(0, 1, len(d)) * 0.05 - 10.0 * (d >= "2020-08-20")
        parts.append(pd.DataFrame({"geo": "IA", "date": d, "season": y, "x": v}))
    return pd.concat(parts, ignore_index=True)


def test_event_timing_uses_prior_season_sigma_only():
    panel = _event_panel()
    z = P.prior_season_z(panel, "IA", 2020, "x")
    prior = panel[(panel["season"] < 2020)]
    day = z.iloc[30]
    vals = prior[prior["date"].dt.strftime("%m-%d") == day["date"].strftime("%m-%d")]["x"]
    assert day["prior_mean"] == pytest.approx(vals.mean())
    assert day["prior_sd"] == pytest.approx(vals.std(ddof=1))
    assert day["n_prior"] == 11
    r = P.first_crossing(panel, "IA", 2020, ["x"], event_date="2020-08-10").iloc[0]
    # later seasons changed wildly: same z and crossing
    p2 = panel.copy()
    p2.loc[p2["season"] > 2020, "x"] *= 100.0
    r2 = P.first_crossing(p2, "IA", 2020, ["x"], event_date="2020-08-10").iloc[0]
    assert r["first_cross"] == r2["first_cross"] and r["z_at_cross"] == r2["z_at_cross"]
    assert r["first_cross"] == pd.Timestamp("2020-08-20") and not r["below_at_event"]
    assert r["first_cross_after_event"] == pd.Timestamp("2020-08-20")
    assert r["days_after_event"] == 10

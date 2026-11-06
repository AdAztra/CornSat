"""Phase 8f: state-panel evaluation of satellite feature groups (S now, HLS "H" later).

Why a state panel
-----------------
HLS (30 m) exists only from 2016, i.e. ~10 seasons: far too few for the national Stage A
model (one row per season). Pooling the 5 core states gives ~5 rows per season. States
within a season share weather and are correlated, so the SEASON stays the unit of
validation and of inference (all splits and all tests group by season).

Target
------
surprise(s, Y) = final state yield(s, Y) - trend_s(Y), where trend_s is an OLS line on
that state's own finals over TREND_START (1990) .. Y-1 only (``model.trend_for_season``).
Finals: NASS state history 1990..2008 (``fetch_state_final_history``, cached to
STATE_FINAL_CACHE) plus the FINAL rows of usda_yield.parquet (2009+). Only finals that
were released (available_date <= ``asof``) are used.

Features
--------
State rows of the as-of panel (features_daily: one row per geo and calendar day; or the
state-suffixed columns of features_trading via ``long_from_trading``), read as they are
(no recomputation) at each SNAPSHOT date: the last row of (state, season) with date <=
snapshot date. Groups (``GROUPS``): W, C, S from ``model.FEATURE_GROUPS`` plus H (HLS),
which is used automatically once its columns exist in the panel (``available_groups``).

Model
-----
Pooled ridge across states, one model per snapshot: standardise (mean / population sd of
the TRAINING rows), ridge without intercept (as Stage A: the prediction shrinks to 0 =
the trend), alpha from ALPHAS chosen by leave-one-SEASON-out (grouped) CV inside the
training rows (``select_alpha``). Optional state demeaning (``demean=True``): per-state
means of target and features are taken from the training rows only, subtracted before
the fit and added back to the prediction (state fixed effects).

Validation
----------
* ``"loso"``: leave-one-season-out; ALL states of the held-out season leave training.
* ``"wf"``: expanding walk-forward; season Y is trained on seasons < Y only, and needs
  at least MIN_TRAIN_SEASONS training seasons.
Rows with any NaN in the set's features get no prediction (no feature dropping).

Comparisons (``compare``)
-------------------------
For each WINDOW the sample is restricted (training AND evaluation) to seasons >= the
window start and to rows complete in every feature of every group compared in that
window plus a target ("strict common sample"), so every set is fit and scored on
identical (state, season, snapshot) rows. Paired squared-error differences
d = se_A - se_B (> 0: B better) are averaged per season (the unit) before the paired
t-test, Wilcoxon test and a season-block bootstrap CI.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from cornsat import model as M
from cornsat.config import DATA_INTERIM, DATA_PROCESSED, STATES
from cornsat.features import cal_doy

log = logging.getLogger(__name__)

STATE_LIST = list(STATES)
MIN_TRAIN_SEASONS = 3
ALPHAS = M.ALPHAS
SNAPSHOTS = M.SNAPSHOTS
TREND_START = M.TREND_START
FEATURES_DAILY_PATH = DATA_PROCESSED / "features_daily.parquet"
FEATURES_TRADING_PATH = DATA_PROCESSED / "features_trading.parquet"
USDA_YIELD_PATH = DATA_PROCESSED / "usda_yield.parquet"
STATE_FINAL_CACHE = DATA_INTERIM / "state_final_yields.parquet"
NASS_CACHE_DIR = DATA_INTERIM / "usda_state_history"  # raw NASS cache for this module

HLS_FEATURES = ["hls_ndvi_z", "hls_evi_anom_cum", "hls_ndvi_trend_anom"]
GROUPS: dict[str, list[str]] = {g: list(M.FEATURE_GROUPS[g]) for g in ("W", "C", "S")}
GROUPS["H"] = list(M.FEATURE_GROUPS.get("H", HLS_FEATURES))

# (A, B): d = se_A - se_B > 0 means B is better.
PAIRS: list[tuple[str, str]] = [
    ("W+C", "W+C+S"), ("W+C", "W+C+H"), ("W+C+S", "W+C+H"), ("S", "H"),
    ("W", "W+S"), ("W", "W+H"), ("C", "C+S"), ("C", "C+H"),
    ("trend", "W+C"), ("trend", "W+C+S"), ("usda", "W+C+S"),
]  # fmt: skip
BASELINE_SETS = ("trend", "usda")
# window label -> (first season, groups compared). H windows start 2016 (HLS start).
WINDOWS: dict[str, tuple[int, tuple[str, ...]]] = {
    "2009+": (2009, ("W", "C", "S")),
    "2016+": (2016, ("W", "C", "S", "H")),
    "2019+": (2019, ("W", "C", "S", "H")),
}
EVENTS = [  # (state, season, event date or None, label)
    ("IA", 2020, "2020-08-10", "derecho"),
    ("IA", 2023, None, "2023 dry summer"),
    ("IL", 2023, None, "2023 dry summer"),
]


def set_features(name: str, groups: dict[str, list[str]] | None = None) -> list[str]:
    """'W+C+S' -> the features of W, C and S (in that order)."""
    groups = GROUPS if groups is None else groups
    return [f for g in name.split("+") for f in groups[g]]


def set_groups(name: str) -> set[str]:
    return set() if name in BASELINE_SETS else set(name.split("+"))


def available_groups(panel: pd.DataFrame, groups: dict[str, list[str]] | None = None) -> list[str]:
    """Groups whose every feature is a column of ``panel`` with at least one value."""
    groups = GROUPS if groups is None else groups
    return [g for g, fs in groups.items()
            if all(f in panel.columns and panel[f].notna().any() for f in fs)]


# --- targets ------------------------------------------------------------------------------
def fetch_state_final_history(first: int = TREND_START, last: int = 2008, states=STATE_LIST,
                              cache=STATE_FINAL_CACHE, refresh: bool = False) -> pd.DataFrame:
    """State final yields first..last (columns state, year, final) from NASS Quick Stats.

    Cached to ``cache``; the raw NASS response is cached by ``usda.query`` under
    NASS_CACHE_DIR (a directory of its own, so the shared usda cache is never written)."""
    from cornsat import usda

    if cache.exists() and not refresh:
        h = pd.read_parquet(cache)
        if set(states) <= set(h["state"]) and h["year"].min() <= first and h["year"].max() >= last:
            return h[(h["year"] >= first) & (h["year"] <= last) & h["state"].isin(states)]
    old = usda.CACHE_DIR
    usda.CACHE_DIR = NASS_CACHE_DIR
    try:
        raw = usda.query(source_desc="SURVEY", commodity_desc="CORN", agg_level_desc="STATE",
                         state_alpha=list(states), statisticcat_desc="YIELD",
                         unit_desc="BU / ACRE", util_practice_desc="GRAIN",
                         year__GE=first, year__LE=last)
    finally:
        usda.CACHE_DIR = old
    y = usda.parse_yield(raw)
    f = y[(y["period"] == "FINAL") & y["geo"].isin(states)]
    h = (f.rename(columns={"geo": "state", "yield_bu_ac": "final"})[["state", "year", "final"]]
         .astype({"year": int, "final": float}).dropna().sort_values(["state", "year"])
         .reset_index(drop=True))
    h = h[(h["year"] >= first) & (h["year"] <= last)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    h.to_parquet(cache, index=False)
    return h


def state_final_yields(usda_yield: pd.DataFrame, history: pd.DataFrame | None = None,
                       asof=None, states=STATE_LIST) -> pd.DataFrame:
    """Final yield by state and year: ``history`` plus the FINAL rows of ``usda_yield``
    released by ``asof`` (default: all rows present; ``usda.parse_yield`` already drops
    unreleased finals). Sample rows win on overlap. Wide: index year, one column per state."""
    u = usda_yield[(usda_yield["period"] == "FINAL") & usda_yield["geo"].isin(states)].copy()
    if asof is not None:
        u = u[pd.to_datetime(u["available_date"]) <= pd.Timestamp(asof)]
    u = u.rename(columns={"geo": "state", "yield_bu_ac": "final"})[["state", "year", "final"]]
    u = u.dropna()
    parts = [u]
    if history is not None and len(history):
        key = set(zip(u["state"], u["year"].astype(int)))
        h = history[[(s, int(y)) not in key for s, y in zip(history["state"], history["year"])]]
        parts.insert(0, h[["state", "year", "final"]])
    long = pd.concat(parts, ignore_index=True).astype({"year": int, "final": float})
    return long.pivot_table(index="year", columns="state", values="final").sort_index()


def state_targets(finals: pd.DataFrame, seasons: Iterable[int], start: int = TREND_START,
                  method: str = "ols") -> pd.DataFrame:
    """Rows (state, season, final, trend, surprise, trend_last): each state's own trend
    from ``model.trend_for_season`` on years start .. season-1 only. ``final`` /
    ``surprise`` are NaN for finals not released."""
    rows = []
    for s in finals.columns:
        f = finals[s].dropna()
        f.index = f.index.astype(int)
        for y in seasons:
            tr, yrs = M.trend_for_season(f, int(y), start, method)
            fin = float(f.get(int(y), np.nan))
            rows.append({"state": s, "season": int(y), "final": fin, "trend": tr,
                         "surprise": fin - tr, "trend_last": max(yrs) if yrs else np.nan})
    return pd.DataFrame(rows)


def usda_state_forecasts(usda_yield: pd.DataFrame, states=STATE_LIST) -> pd.DataFrame:
    """Released state forecasts (AUG..NOV): state, year, yield_bu_ac, available_date."""
    f = usda_yield[(usda_yield["period"] != "FINAL") & usda_yield["geo"].isin(states)
                   & usda_yield["yield_bu_ac"].notna()]
    f = f.rename(columns={"geo": "state"})[["state", "year", "yield_bu_ac", "available_date"]]
    f = f.assign(available_date=pd.to_datetime(f["available_date"]))
    return f.sort_values(["state", "year", "available_date"]).reset_index(drop=True)


# --- features -----------------------------------------------------------------------------
def long_from_trading(trading: pd.DataFrame, states=STATE_LIST) -> pd.DataFrame:
    """features_trading (state columns suffixed _ia, _il, ...) -> long rows (geo, date,
    season, doy, <unsuffixed features>)."""
    t = trading.copy()
    t["date"] = pd.to_datetime(t["date"])
    season = t["campaign"].astype(int) if "campaign" in t else t["date"].dt.year
    parts = []
    for s in states:
        suf = f"_{s.lower()}"
        cols = {c: c[: -len(suf)] for c in t.columns if c.endswith(suf)}
        p = t[list(cols)].rename(columns=cols)
        p.insert(0, "geo", s)
        p.insert(1, "date", t["date"])
        p.insert(2, "season", season)
        parts.append(p)
    out = pd.concat(parts, ignore_index=True)
    out["doy"] = cal_doy(out["date"])
    return out.sort_values(["geo", "date"]).reset_index(drop=True)


def load_panel(source: str = "daily") -> pd.DataFrame:
    """State rows of the as-of panel: ``"daily"`` (features_daily) or ``"trading"``."""
    if source == "daily":
        p = pd.read_parquet(FEATURES_DAILY_PATH)
        p = p[p["geo"].isin(STATE_LIST)].copy()
        p["date"] = pd.to_datetime(p["date"])
        p["season"] = p["season"].astype(int)
        return p.sort_values(["geo", "date"]).reset_index(drop=True)
    if source == "trading":
        return long_from_trading(pd.read_parquet(FEATURES_TRADING_PATH))
    raise ValueError(source)


def snapshot_features(panel: pd.DataFrame, features: Sequence[str], snapshots=SNAPSHOTS,
                      states=STATE_LIST) -> pd.DataFrame:
    """One row per (state, season, snapshot): the panel's last row of that state and
    season with date <= the snapshot date, as is. Columns state, season, snapshot,
    snap_date (the snapshot calendar date), date (row used), features (NaN if absent)."""
    p = panel[panel["geo"].isin(states)]
    feats = [f for f in features if f in p.columns]
    rows = []
    for (s, y), g in p.groupby(["geo", "season"], sort=True):
        dates = g["date"].to_numpy()
        for label, (m, d) in snapshots.items():
            sd = pd.Timestamp(int(y), m, d)
            k = np.searchsorted(dates, np.datetime64(sd), side="right")
            if k == 0:
                continue
            r = g.iloc[k - 1]
            rows.append({"state": s, "season": int(y), "snapshot": label, "snap_date": sd,
                         "date": pd.Timestamp(r["date"]), **{f: r[f] for f in feats}})
    out = pd.DataFrame(rows)
    for f in features:
        if f not in out:
            out[f] = np.nan
    return out


def build_dataset(panel: pd.DataFrame, targets: pd.DataFrame, forecasts: pd.DataFrame | None = None,
                  groups: dict[str, list[str]] | None = None, snapshots=SNAPSHOTS) -> pd.DataFrame:
    """Snapshot features + targets (+ ``usda``: latest state forecast released by the
    snapshot row's date, minus the state trend; NaN before the August report)."""
    groups = GROUPS if groups is None else groups
    feats = list(dict.fromkeys(f for fs in groups.values() for f in fs))
    d = snapshot_features(panel, feats, snapshots)
    d = d.merge(targets[["state", "season", "final", "trend", "surprise"]],
                on=["state", "season"], how="left")
    d["usda"] = np.nan
    if forecasts is not None and len(forecasts):
        fc = {k: g for k, g in forecasts.groupby(["state", "year"])}
        vals = []
        for s, y, dt, tr in zip(d["state"], d["season"], d["date"], d["trend"]):
            g = fc.get((s, y))
            g = None if g is None else g[g["available_date"] <= dt]
            vals.append(np.nan if g is None or g.empty else float(g["yield_bu_ac"].iloc[-1] - tr))
        d["usda"] = vals
    return d


# --- pooled ridge -------------------------------------------------------------------------
class PooledRidge:
    """Standardise + ridge (no intercept), optionally after per-state demeaning. Every
    statistic (state means, scaler mean / sd, coefficients) comes from the rows passed to
    ``fit``."""

    def __init__(self, alpha: float, demean: bool = False):
        self.alpha = float(alpha)
        self.demean = demean

    def fit(self, X: np.ndarray, y: np.ndarray, state: np.ndarray | None = None) -> PooledRidge:
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.state_x_, self.state_y_ = {}, {}
        self.xbar_all_ = X.mean(0)
        if self.demean:
            state = np.asarray(state)
            X, y = X.copy(), y.copy()
            for s in np.unique(state):
                m = state == s
                self.state_x_[s], self.state_y_[s] = X[m].mean(0), y[m].mean()
                X[m] -= self.state_x_[s]
                y[m] -= self.state_y_[s]
        self.mean_ = X.mean(0)
        sd = X.std(0)
        self.scale_ = np.where(sd > 1e-12, sd, 1.0)  # StandardScaler convention
        Z = (X - self.mean_) / self.scale_
        k = Z.shape[1]
        self.coef_ = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(k), Z.T @ y)
        return self

    def predict(self, X: np.ndarray, state: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, float)
        off = np.zeros(len(X))
        if self.demean:
            state = np.asarray(state)
            X = X.copy()
            for i, s in enumerate(state):
                if s in self.state_x_:
                    X[i] = X[i] - self.state_x_[s]
                    off[i] = self.state_y_[s]
                else:  # unseen state: pooled training mean, no fixed effect (0 = trend)
                    X[i] = X[i] - self.xbar_all_
        return off + ((X - self.mean_) / self.scale_) @ self.coef_


def select_alpha(X: np.ndarray, y: np.ndarray, season: np.ndarray, state: np.ndarray | None = None,
                 alphas=ALPHAS, demean: bool = False) -> float:
    """Alpha minimising the pooled squared error of leave-one-SEASON-out CV inside the
    given (training) rows: each fold removes every row of one season. Ties -> smaller
    alpha. Fewer than 2 seasons -> the median alpha."""
    season = np.asarray(season)
    us = np.unique(season)
    if len(us) < 2:
        return float(sorted(alphas)[len(alphas) // 2])
    sse = []
    for a in alphas:
        tot = 0.0
        for s in us:
            tr, te = season != s, season == s
            m = PooledRidge(a, demean).fit(X[tr], y[tr], None if state is None else state[tr])
            tot += float(((m.predict(X[te], None if state is None else state[te]) - y[te]) ** 2).sum())
        sse.append(tot)
    return float(alphas[int(np.argmin(sse))])


def training_seasons(scheme: str, season: int, seasons: Iterable[int]) -> list[int]:
    """"wf": seasons < season; "loso": seasons != season."""
    s = sorted({int(x) for x in seasons})
    if scheme == "wf":
        return [x for x in s if x < season]
    if scheme == "loso":
        return [x for x in s if x != season]
    raise ValueError(scheme)


def cv_predictions(data: pd.DataFrame, features: Sequence[str], scheme: str,
                   alphas=ALPHAS, demean: bool = False,
                   min_train_seasons: int = MIN_TRAIN_SEASONS, target: str = "surprise") -> pd.DataFrame:
    """Out-of-sample predictions, one pooled model per (snapshot, held-out season).

    Only rows complete in ``features`` and ``target`` are used, for training and as
    targets (strict: no feature dropping). Training = complete rows of the same snapshot
    in ``training_seasons(scheme, Y, ...)``; alpha by grouped LOSO inside them. Returns
    rows state, season, snapshot, pred, actual, alpha, n_train, n_train_seasons,
    train_last (newest training season)."""
    feats = list(features)
    ok = data.dropna(subset=[*feats, target])
    out = []
    for snap, g in ok.groupby("snapshot", sort=False):
        X, y = g[feats].to_numpy(float), g[target].to_numpy(float)
        seas, st = g["season"].to_numpy(int), g["state"].to_numpy()
        for Y in sorted(set(seas)):
            pool = training_seasons(scheme, Y, seas)
            if len(pool) < min_train_seasons:
                continue
            tr, te = np.isin(seas, pool), seas == Y
            a = select_alpha(X[tr], y[tr], seas[tr], st[tr], alphas, demean)
            m = PooledRidge(a, demean).fit(X[tr], y[tr], st[tr])
            p = m.predict(X[te], st[te])
            for s, pv, av in zip(st[te], p, y[te]):
                out.append({"state": s, "season": int(Y), "snapshot": snap, "pred": float(pv),
                            "actual": float(av), "alpha": a, "n_train": int(tr.sum()),
                            "n_train_seasons": len(pool), "train_last": max(pool)})
    return pd.DataFrame(out, columns=["state", "season", "snapshot", "pred", "actual", "alpha",
                                      "n_train", "n_train_seasons", "train_last"])


# --- comparisons --------------------------------------------------------------------------
def common_sample(data: pd.DataFrame, groups: Iterable[str], first_season: int,
                  group_map: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Rows of seasons >= first_season complete in every feature of ``groups`` and in
    the target: the identical sample on which all sets of a window are fit and scored."""
    feats = [f for g in groups for f in (GROUPS if group_map is None else group_map)[g]]
    d = data[data["season"] >= first_season]
    return d.dropna(subset=[*feats, "surprise"]).reset_index(drop=True)


def window_predictions(sample: pd.DataFrame, sets: Sequence[str], schemes=("loso", "wf"),
                       alphas=ALPHAS, demean: bool = False,
                       min_train_seasons: int = MIN_TRAIN_SEASONS,
                       group_map: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Predictions of every model set and baseline on ``sample`` (a ``common_sample``).
    Baselines are scored on the rows the model sets predict for the same scheme
    (``trend`` = 0; ``usda`` = released state forecast surprise, NaN before August)."""
    parts = []
    for scheme in schemes:
        keys = None
        for s in sets:
            if s in BASELINE_SETS:
                continue
            p = cv_predictions(sample, set_features(s, group_map), scheme, alphas, demean,
                               min_train_seasons).assign(scheme=scheme, set=s)
            parts.append(p)
            keys = p[["state", "season", "snapshot"]] if keys is None else keys
        if keys is None:
            continue
        b = keys.merge(sample[["state", "season", "snapshot", "surprise", "usda"]],
                       on=["state", "season", "snapshot"], how="left")
        for s in BASELINE_SETS:
            if s in sets:
                pred = 0.0 if s == "trend" else b["usda"]
                parts.append(b[["state", "season", "snapshot"]].assign(
                    pred=pred, actual=b["surprise"], scheme=scheme, set=s))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _snap_order(c: pd.Series) -> pd.Series:
    order = {k: i for i, k in enumerate([*SNAPSHOTS, "ALL"])}
    return c.map(lambda v: order.get(v, 99))


def accuracy(preds: pd.DataFrame) -> pd.DataFrame:
    """RMSE / MAE / bias / corr per scheme x set x snapshot (+ "ALL" snapshots pooled)."""
    p = preds.dropna(subset=["pred", "actual"])
    out = []
    for (sch, s), g0 in p.groupby(["scheme", "set"], sort=False):
        for snap, g in [*g0.groupby("snapshot", sort=False), ("ALL", g0)]:
            e = g["pred"] - g["actual"]
            corr = (np.corrcoef(g["pred"], g["actual"])[0, 1]
                    if len(g) > 2 and g["pred"].std() > 1e-12 else np.nan)
            out.append({"scheme": sch, "set": s, "snapshot": snap, "n_rows": len(g),
                        "n_seasons": g["season"].nunique(),
                        "seasons": f"{g['season'].min()}-{g['season'].max()}",
                        "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(e.abs().mean()),
                        "bias": float(e.mean()), "corr": corr})
    df = pd.DataFrame(out)
    return df.sort_values(["scheme", "set", "snapshot"], key=lambda c: _snap_order(c)
                          if c.name == "snapshot" else c).reset_index(drop=True) if len(df) else df


def by_state(preds: pd.DataFrame) -> pd.DataFrame:
    """RMSE per scheme x set x state, all snapshots pooled."""
    p = preds.dropna(subset=["pred", "actual"])
    g = p.assign(se=(p["pred"] - p["actual"]) ** 2).groupby(["scheme", "set", "state"])
    return g.agg(n_rows=("se", "size"), n_seasons=("season", "nunique"),
                 rmse=("se", lambda v: float(np.sqrt(v.mean())))).reset_index()


def season_block_bootstrap(d_by_season: np.ndarray, n_boot: int = 5000, seed: int = 0,
                           level: float = 0.95) -> tuple[float, float]:
    """Percentile CI of the mean of per-season values, resampling SEASONS."""
    d = np.asarray(d_by_season, float)
    if len(d) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    q = (1 - level) / 2
    return float(np.quantile(means, q)), float(np.quantile(means, 1 - q))


def paired(preds: pd.DataFrame, pairs=PAIRS, n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Paired squared-error comparison A vs B on the rows where both predict (identical
    (state, season, snapshot) keys). d = se_A - se_B (> 0: B better). Per snapshot and
    "ALL"; d is averaged per season first (season = unit) for mean, t, Wilcoxon and
    the season-block bootstrap CI."""
    p = preds.dropna(subset=["pred", "actual"])
    p = p.assign(se=(p["pred"] - p["actual"]) ** 2)
    out = []
    for sch, ps in p.groupby("scheme", sort=False):
        for a, b in pairs:
            w = ps[ps["set"].isin([a, b])].pivot_table(
                index=["snapshot", "season", "state"], columns="set", values="se")
            if a not in w or b not in w:
                continue
            w = w[[a, b]].dropna()
            if w.empty:
                continue
            w["d"] = w[a] - w[b]
            groups = [*w.groupby(level="snapshot", sort=False), ("ALL", w)]
            for snap, g in groups:
                ds = g.groupby(level="season")["d"].mean().to_numpy()
                n = len(ds)
                t_p = stats.ttest_1samp(ds, 0.0).pvalue if n > 2 and ds.std() > 0 else np.nan
                try:
                    w_p = stats.wilcoxon(ds).pvalue if n > 2 and np.any(ds != 0) else np.nan
                except ValueError:
                    w_p = np.nan
                lo, hi = season_block_bootstrap(ds, n_boot, seed)
                out.append({"scheme": sch, "A": a, "B": b, "snapshot": snap, "n_rows": len(g),
                            "n_seasons": n, "rmse_A": float(np.sqrt(g[a].mean())),
                            "rmse_B": float(np.sqrt(g[b].mean())), "mean_d": float(ds.mean()),
                            "n_seasons_B_better": int((ds > 0).sum()), "p_ttest": t_p,
                            "p_wilcoxon": w_p, "boot_lo": lo, "boot_hi": hi})
    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df.sort_values(["scheme", "A", "B", "snapshot"], key=lambda c: _snap_order(c)
                          if c.name == "snapshot" else c).reset_index(drop=True)


def compare(data: pd.DataFrame, windows=WINDOWS, pairs=PAIRS, schemes=("loso", "wf"),
            alphas=ALPHAS, demean: bool = False, min_train_seasons: int = MIN_TRAIN_SEASONS,
            group_map: dict[str, list[str]] | None = None, n_boot: int = 5000):
    """All windows: (predictions, accuracy, paired, by_state, skipped) with a ``window``
    column. A window uses the groups of WINDOWS that are available in ``data``; pairs
    needing a missing group are listed in ``skipped``."""
    gm = GROUPS if group_map is None else group_map
    avail = set(available_groups(data, gm))
    P, A, D, S, skipped = [], [], [], [], []
    for label, (first, wgroups) in windows.items():
        groups = [g for g in wgroups if g in avail]
        ok_pairs = [(a, b) for a, b in pairs if (set_groups(a) | set_groups(b)) <= set(groups)]
        skipped += [{"window": label, "A": a, "B": b,
                     "reason": f"group(s) {sorted((set_groups(a) | set_groups(b)) - set(groups))} "
                               "not available"} for a, b in pairs if (a, b) not in ok_pairs]
        if not ok_pairs:
            continue
        sample = common_sample(data, groups, first, gm)
        sets = list(dict.fromkeys(s for ab in ok_pairs for s in ab))
        log.info("window %s: groups %s, %d rows, %d seasons", label, groups, len(sample),
                 sample["season"].nunique())
        preds = window_predictions(sample, sets, schemes, alphas, demean, min_train_seasons, gm)
        if preds.empty:
            continue
        preds = preds.assign(window=label)
        P.append(preds)
        A.append(accuracy(preds).assign(window=label, groups="+".join(groups)))
        D.append(paired(preds, ok_pairs, n_boot).assign(window=label))
        S.append(by_state(preds).assign(window=label))

    def cat(x):
        if not x:
            return pd.DataFrame()
        df = pd.concat(x, ignore_index=True)
        return df[["window", *[c for c in df.columns if c != "window"]]]

    return cat(P), cat(A), cat(D), cat(S), pd.DataFrame(skipped)


# --- event timing -------------------------------------------------------------------------
def prior_season_z(panel: pd.DataFrame, state: str, season: int, feature: str,
                   min_prior: int = 3) -> pd.DataFrame:
    """Season ``season`` of ``state``: the feature standardised per day of year by the
    mean / sd (ddof=1) of that feature on the same DOY in PRIOR seasons only (< season).
    Columns date, doy, value, prior_mean, prior_sd, n_prior, z."""
    p = panel[(panel["geo"] == state) & panel[feature].notna()] if feature in panel else panel.iloc[:0]
    p = p.assign(doy=cal_doy(p["date"]))
    cur = p[p["season"] == season][["date", "doy", feature]].rename(columns={feature: "value"})
    prior = p[p["season"] < season].groupby("doy")[feature].agg(["mean", "std", "count"])
    prior.columns = ["prior_mean", "prior_sd", "n_prior"]
    out = cur.merge(prior, left_on="doy", right_index=True, how="left")
    ok = (out["n_prior"] >= min_prior) & (out["prior_sd"] > 0)
    out["z"] = np.where(ok, (out["value"] - out["prior_mean"]) / out["prior_sd"], np.nan)
    return out.sort_values("date").reset_index(drop=True)


def first_crossing(panel: pd.DataFrame, state: str, season: int, features: Sequence[str],
                   threshold: float = -1.0, start=None, end=None, event_date=None,
                   min_prior: int = 3) -> pd.DataFrame:
    """Per feature: first date in [start, end] (default Jun 1 .. Oct 31) on which the
    prior-season z (``prior_season_z``) is <= threshold, and the first such date on or
    after ``event_date`` (when given), with lead/lag days relative to the event."""
    start = pd.Timestamp(season, 6, 1) if start is None else pd.Timestamp(start)
    end = pd.Timestamp(season, 10, 31) if end is None else pd.Timestamp(end)
    ev = None if event_date is None else pd.Timestamp(event_date)
    rows = []
    for f in features:
        z = prior_season_z(panel, state, season, f, min_prior)
        z = z[(z["date"] >= start) & (z["date"] <= end)]
        hit = z[z["z"] <= threshold]
        r = {"state": state, "season": season, "feature": f, "threshold": threshold,
             "event_date": ev, "n_days": int(z["z"].notna().sum()),
             "n_prior": int(z["n_prior"].max()) if len(z) and z["n_prior"].notna().any() else 0,
             "first_cross": hit["date"].iloc[0] if len(hit) else pd.NaT,
             "z_at_cross": float(hit["z"].iloc[0]) if len(hit) else np.nan,
             "min_z": float(z["z"].min()) if z["z"].notna().any() else np.nan,
             "min_z_date": z.loc[z["z"].idxmin(), "date"] if z["z"].notna().any() else pd.NaT}
        if ev is not None:
            # a NEW crossing on/after the event: z <= threshold with z > threshold on the
            # previous observed day (a series already below at the event does not count)
            zz = z[z["z"].notna()]
            below = zz["z"] <= threshold
            after = zz[below & ~below.shift(1, fill_value=False) & (zz["date"] >= ev)]
            pre = z[(z["date"] < ev) & z["z"].notna()]
            r["z_day_before_event"] = float(pre["z"].iloc[-1]) if len(pre) else np.nan
            r["below_at_event"] = bool(len(pre) and pre["z"].iloc[-1] <= threshold)
            r["first_cross_after_event"] = after["date"].iloc[0] if len(after) else pd.NaT
            r["days_after_event"] = (float((after["date"].iloc[0] - ev).days)
                                     if len(after) else np.nan)
        rows.append(r)
    return pd.DataFrame(rows)


def event_table(panel: pd.DataFrame, events=EVENTS, groups=("S", "C", "H"),
                group_map: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """``first_crossing`` for each event and the features of the available ``groups``."""
    gm = GROUPS if group_map is None else group_map
    avail = available_groups(panel, gm)
    feats = [f for g in groups if g in avail for f in gm[g]]
    out = [first_crossing(panel, s, y, feats, event_date=ev).assign(event=lab)
           for s, y, ev, lab in events]
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

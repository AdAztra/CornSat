"""Phase 6, Stage A: US corn yield-surprise model (see reports/phase6_design.md).

Target
------
surprise(Y) = US final yield(Y) - trend(Y), where trend(Y) is a linear fit (OLS by
default, or Huber) of US final yield on year over TREND_START .. Y-1 only
(``trend_for_season``). Every season's target uses its own trend, so a training target
never contains information from its own season or later.

Features
--------
Rows of ``features_trading`` (one per ZCZ trading date; row d is as of the morning of d).
FEATURE_GROUPS lists the W (weather), C (crop conditions) and S (satellite) columns.
FEATURE_SETS are the groups and their combinations.

Model
-----
StandardScaler + RidgeCV without an intercept, fit on training seasons only
(``fit_ridge``). Centred features leave the slopes unchanged. Without an intercept the
prediction shrinks to 0, the trend, as alpha grows. Alpha is chosen by efficient
leave-one-out over ALPHAS, inside the training seasons only. Features that are NaN in the
target row are dropped from that prediction (``fallback``). Training seasons with NaN in
a used feature are dropped. With fewer than MIN_TRAIN training seasons there is no
prediction (NaN).

Validation
----------
* ``"wf"``: expanding walk-forward. Season Y is trained on seasons < Y only. This is the
  honest scheme and the only one used for trading.
* ``"loso"``: leave-one-season-out. Trains on future seasons; for comparison only.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cornsat.config import DATA_PROCESSED
from cornsat.features import cal_doy

log = logging.getLogger(__name__)

TREND_START = 1990
FIRST_SEASON = 2009
MIN_TRAIN = 5
ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)

FEATURE_GROUPS: dict[str, list[str]] = {
    "W": ["edd30_cum", "precip_30d_anom", "gdd_cum_anom"],
    "C": ["ge", "ge_chg_2w"],
    "S": ["ndvi_z", "evi_anom_cum", "ndvi_trend_anom"],
}
SET_NAMES = ["W", "C", "S", "W+C", "W+S", "C+S", "W+C+S"]
FEATURE_SETS: dict[str, list[str]] = {
    name: [f for g in name.split("+") for f in FEATURE_GROUPS[g]] for name in SET_NAMES
}
S_FEATURES = set(FEATURE_GROUPS["S"])

SNAPSHOTS: dict[str, tuple[int, int]] = {
    "Jun15": (6, 15), "Jul01": (7, 1), "Jul15": (7, 15), "Aug01": (8, 1),
    "Aug15": (8, 15), "Sep01": (9, 1), "Sep15": (9, 15),
}  # fmt: skip
BASELINES = ["trend", "mean", "usda"]
PREDICTIONS_PATH = DATA_PROCESSED / "model_predictions.parquet"


# --- yields and trend ----------------------------------------------------------------
def load_us_yields() -> pd.DataFrame:
    """usda_yield.parquet rows for geo US."""
    y = pd.read_parquet(DATA_PROCESSED / "usda_yield.parquet")
    y = y[y["geo"] == "US"].copy()
    y["available_date"] = pd.to_datetime(y["available_date"])
    return y.reset_index(drop=True)


def fetch_final_history(first: int = TREND_START, last: int = FIRST_SEASON - 1) -> pd.Series:
    """US final yields first..last from NASS Quick Stats (``usda.query``, cached)."""
    from cornsat import usda

    raw = usda.query(source_desc="SURVEY", commodity_desc="CORN", agg_level_desc="NATIONAL",
                     statisticcat_desc="YIELD", unit_desc="BU / ACRE",
                     util_practice_desc="GRAIN", year__GE=first, year__LE=last)
    y = usda.parse_yield(raw)
    f = y[(y["geo"] == "US") & (y["period"] == "FINAL")]
    s = f.set_index("year")["yield_bu_ac"].astype(float)
    return s[(s.index >= first) & (s.index <= last)].sort_index()


def us_final_yields(us_yields: pd.DataFrame, history: pd.Series | None = None) -> pd.Series:
    """US final yield by year: ``history`` (years before the sample) plus FINAL rows of
    ``us_yields``. Sample rows win on overlap."""
    f = us_yields[us_yields["period"] == "FINAL"].set_index("year")["yield_bu_ac"].astype(float)
    f = f.dropna()
    if history is not None:
        f = pd.concat([history[~history.index.isin(f.index)], f])
    f.index = f.index.astype(int)
    return f.sort_index().rename("final")


def trend_for_season(final: pd.Series, season: int, start: int = TREND_START,
                     method: str = "ols") -> tuple[float, list[int]]:
    """Trend yield for ``season`` from a linear fit on years start .. season-1 ONLY.

    Returns (trend value at ``season``, years used). ``method``: "ols" or "huber"
    (statsmodels RLM with HuberT)."""
    hist = final[(final.index >= start) & (final.index < season)].dropna()
    years = [int(y) for y in hist.index]
    if len(hist) < 3:
        return np.nan, years
    X = sm.add_constant(hist.index.to_numpy(float))
    if method == "ols":
        params = sm.OLS(hist.to_numpy(float), X).fit().params
    elif method == "huber":
        params = sm.RLM(hist.to_numpy(float), X, M=sm.robust.norms.HuberT()).fit().params
    else:
        raise ValueError(method)
    return float(params[0] + params[1] * season), years


def season_targets(final: pd.Series, seasons: Iterable[int], start: int = TREND_START,
                   method: str = "ols") -> pd.DataFrame:
    """Per season: final (NaN if not yet released), trend, surprise = final - trend."""
    rows = []
    for y in seasons:
        tr, yrs = trend_for_season(final, y, start, method)
        fin = float(final.get(y, np.nan))
        rows.append({"season": int(y), "final": fin, "trend": tr, "surprise": fin - tr,
                     "trend_first": min(yrs) if yrs else np.nan,
                     "trend_last": max(yrs) if yrs else np.nan})
    return pd.DataFrame(rows).set_index("season")


def us_forecasts(us_yields: pd.DataFrame) -> pd.DataFrame:
    """Released, non-NaN US forecasts (AUG..NOV) with their available_date."""
    f = us_yields[(us_yields["period"] != "FINAL") & us_yields["yield_bu_ac"].notna()]
    return f[["year", "period", "yield_bu_ac", "available_date"]].sort_values(
        ["year", "available_date"]).reset_index(drop=True)


def usda_surprise_at(forecasts: pd.DataFrame, targets: pd.DataFrame, season: int,
                     date) -> float:
    """Latest US forecast for ``season`` with available_date <= date, minus trend(season);
    NaN before the first (August) forecast is usable."""
    f = forecasts[(forecasts["year"] == season) & (forecasts["available_date"] <= pd.Timestamp(date))]
    if f.empty:
        return np.nan
    return float(f.iloc[-1]["yield_bu_ac"] - targets.loc[season, "trend"])


# --- feature rows ---------------------------------------------------------------------
def prepare_features(feat: pd.DataFrame) -> pd.DataFrame:
    """features_trading with ``season`` (= campaign), ``doy`` (365-day calendar)."""
    f = feat.copy()
    f["date"] = pd.to_datetime(f["date"])
    f["season"] = f["campaign"].astype(int)
    f["doy"] = cal_doy(f["date"])
    return f.sort_values("date").reset_index(drop=True)


def snapshot_rows(feat: pd.DataFrame, month: int, day: int) -> pd.DataFrame:
    """Per season, the row of the first trading day on or after (month, day); index season."""
    rows = {}
    for y, g in feat.groupby("season"):
        g = g[g["date"] >= pd.Timestamp(int(y), month, day)]
        if len(g):
            rows[int(y)] = g.iloc[0]
    return pd.DataFrame.from_dict(rows, orient="index")


class AsofIndex:
    """Fast lookup of each season's last row with doy <= a given doy (same day of season
    in another year)."""

    def __init__(self, feat: pd.DataFrame):
        self.feat = feat
        self.pos = {int(y): (g["doy"].to_numpy(), g.index.to_numpy())
                    for y, g in feat.groupby("season")}

    def rows(self, seasons: Iterable[int], doy: int) -> pd.DataFrame:
        idx, keep = [], []
        for y in seasons:
            if y not in self.pos:
                continue
            doys, ix = self.pos[y]
            k = np.searchsorted(doys, doy, side="right")
            if k > 0:
                idx.append(ix[k - 1])
                keep.append(int(y))
        out = self.feat.loc[idx].copy()
        out.index = keep
        return out


# --- model ------------------------------------------------------------------------------
def fit_ridge(X: pd.DataFrame, y: pd.Series, alphas=ALPHAS) -> Pipeline:
    """StandardScaler + RidgeCV (no intercept, efficient LOO over ``alphas``) fit on X, y
    (training seasons only)."""
    model = Pipeline([("scaler", StandardScaler()),
                      ("ridge", RidgeCV(alphas=list(alphas), fit_intercept=False))])
    model.fit(X.astype(float), y.astype(float))
    return model


def training_pool(scheme: str, season: int, seasons_with_target: Iterable[int]) -> list[int]:
    """Seasons allowed in training: "wf" -> y < season; "loso" -> y != season."""
    s = [int(y) for y in seasons_with_target]
    if scheme == "wf":
        return [y for y in s if y < season]
    if scheme == "loso":
        return [y for y in s if y != season]
    raise ValueError(scheme)


def predict_one(train_X: pd.DataFrame, train_y: pd.Series, x: pd.Series, features: list[str],
                alphas=ALPHAS, min_train: int = MIN_TRAIN) -> dict:
    """Fit on (train_X, train_y) (rows indexed by season) and predict the target row x.

    Fallback rules, applied in order:
    1. Features NaN in x are dropped.
    2. While fewer than ``min_train`` training seasons have every used feature and a
       target, drop the used feature with the most missing training values. This
       happens early in June, when ``ge_chg_2w`` is not yet reported in some years.
    3. If no feature is left, or fewer than ``min_train`` seasons remain, the prediction
       is NaN.
    ``fallback``: some feature of the set was dropped. ``s_dropped``: the set has S
    features and none of them was used, so the model became the one without S (2025 after
    Aug 10)."""
    used = [f for f in features if pd.notna(x.get(f, np.nan))]
    ty = train_y.rename("_y").astype(float).dropna()
    tX = train_X.loc[train_X.index.isin(ty.index)]
    while used:
        d = tX[used].astype(float).join(ty, how="inner").dropna()
        if len(d) >= min_train:
            break
        miss = tX[used].isna().sum()
        used.remove(miss.idxmax())
    has_s = any(f in S_FEATURES for f in features)
    out = {"pred": np.nan, "used": ",".join(used), "fallback": len(used) < len(features),
           "s_dropped": has_s and not any(f in S_FEATURES for f in used),
           "n_train": 0, "train_seasons": [], "alpha": np.nan}
    if not used:
        return out
    model = fit_ridge(d[used], d["_y"], alphas)
    xt = pd.DataFrame([x[used].astype(float).to_numpy()], columns=used)
    out.update(pred=float(model.predict(xt)[0]), n_train=len(d),
               train_seasons=[int(s) for s in d.index], alpha=float(model["ridge"].alpha_))
    return out


def _with_targets(targets: pd.DataFrame) -> list[int]:
    return [int(y) for y in targets.index[targets["surprise"].notna()]]


def snapshot_predictions(feat: pd.DataFrame, targets: pd.DataFrame, forecasts: pd.DataFrame,
                         schemes=("wf", "loso"), snapshots=SNAPSHOTS, sets=SET_NAMES,
                         alphas=ALPHAS, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """Stage A predictions at the fixed snapshots for every scheme, feature set and
    baseline. ``feat`` from ``prepare_features``. Walk-forward rows exist only for seasons
    with at least ``min_train`` prior seasons; LOSO rows for seasons with a target."""
    with_t = _with_targets(targets)
    rows = []
    for label, (m, d) in snapshots.items():
        snap = snapshot_rows(feat, m, d)
        for scheme in schemes:
            if scheme == "wf":
                seasons = [y for y in snap.index if len(training_pool("wf", y, with_t)) >= min_train]
            else:
                seasons = [y for y in snap.index if y in with_t]
            for Y in seasons:
                pool = [y for y in training_pool(scheme, Y, with_t) if y in snap.index]
                tX, ty = snap.loc[pool], targets.loc[pool, "surprise"]
                x = snap.loc[Y]
                base = {"scheme": scheme, "snapshot": label, "season": int(Y),
                        "date": pd.Timestamp(x["date"]),
                        "actual": float(targets.loc[Y, "surprise"]) if Y in targets.index else np.nan}
                for s in sets:
                    r = predict_one(tX, ty, x, FEATURE_SETS[s], alphas, min_train)
                    rows.append({**base, "set": s, **r})
                bl = {"trend": 0.0, "mean": float(ty.mean()),
                      "usda": usda_surprise_at(forecasts, targets, Y, x["date"])}
                for s, v in bl.items():
                    rows.append({**base, "set": s, "pred": v, "used": "", "fallback": False,
                                 "s_dropped": False, "n_train": len(pool) if s == "mean" else 0,
                                 "train_seasons": pool if s == "mean" else [],
                                 "alpha": np.nan})
    return pd.DataFrame(rows)


def rebalance_predictions(feat: pd.DataFrame, targets: pd.DataFrame, dates: pd.DataFrame,
                          sets=("W", "C", "W+C", "W+C+S", "S"), alphas=ALPHAS,
                          min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """Walk-forward predictions refit at each (season, date) of ``dates``.

    Training rows: each season y < Y (with a target) at its own last row with doy <= the
    doy of ``date`` (the same day of season). The target row is ``feat``'s row on
    ``date`` (must exist). Only rows <= date of season Y and rows of seasons < Y are
    read."""
    with_t = _with_targets(targets)
    ax = AsofIndex(feat)
    by_date = feat.set_index("date")
    rows = []
    for Y, d in zip(dates["season"].astype(int), pd.to_datetime(dates["date"])):
        x = by_date.loc[d]
        if isinstance(x, pd.DataFrame):
            x = x.iloc[-1]
        pool = training_pool("wf", Y, with_t)
        tX = ax.rows(pool, int(x["doy"]))
        ty = targets.loc[tX.index, "surprise"]
        for s in sets:
            r = predict_one(tX, ty, x, FEATURE_SETS[s], alphas, min_train)
            rows.append({"scheme": "wf", "season": Y, "date": d, "set": s, **r})
    return pd.DataFrame(rows)


# --- evaluation -------------------------------------------------------------------------
def accuracy_table(preds: pd.DataFrame) -> pd.DataFrame:
    """RMSE / MAE / correlation per scheme x snapshot x set, on seasons with a target."""
    p = preds.dropna(subset=["pred", "actual"])
    out = []
    for (sch, snap, s), g in p.groupby(["scheme", "snapshot", "set"], sort=False):
        e = g["pred"] - g["actual"]
        corr = (np.corrcoef(g["pred"], g["actual"])[0, 1]
                if len(g) > 2 and g["pred"].std() > 1e-12 else np.nan)
        out.append({"scheme": sch, "snapshot": snap, "set": s, "n": len(g),
                    "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(e.abs().mean()),
                    "corr": corr, "n_fallback": int(g["fallback"].sum()),
                    "seasons": f"{g['season'].min()}-{g['season'].max()}"})
    return pd.DataFrame(out)


def paired_comparison(preds: pd.DataFrame, pairs=(("W+C", "W+C+S"), ("W", "W+S"), ("C", "C+S")),
                      scheme: str = "wf") -> pd.DataFrame:
    """Paired squared-error comparison, base (A) vs with satellite (B), per snapshot and
    pooled ("ALL": per-season mean over snapshots, so n stays the number of seasons).
    d = e_A^2 - e_B^2 (> 0 means S helped)."""
    p = preds[(preds["scheme"] == scheme)].dropna(subset=["pred", "actual"])
    p = p.assign(se=(p["pred"] - p["actual"]) ** 2)
    out = []
    for a, b in pairs:
        w = p[p["set"].isin([a, b])].pivot_table(index=["snapshot", "season"], columns="set",
                                                 values="se").dropna()
        if w.empty:
            continue
        w["d"] = w[a] - w[b]
        groups = [(snap, g.droplevel(0)) for snap, g in w.groupby(level=0, sort=False)]
        allg = w.groupby(level="season")[[a, b, "d"]].mean()
        for snap, g in [*groups, ("ALL", allg)]:
            dd = g["d"].to_numpy()
            n = len(dd)
            t_p = stats.ttest_1samp(dd, 0.0).pvalue if n > 2 and dd.std() > 0 else np.nan
            try:
                w_p = stats.wilcoxon(dd).pvalue if n > 2 and np.any(dd != 0) else np.nan
            except ValueError:
                w_p = np.nan
            out.append({"scheme": scheme, "base": a, "with_s": b, "snapshot": snap, "n": n,
                        "rmse_base": float(np.sqrt(g[a].mean())),
                        "rmse_with_s": float(np.sqrt(g[b].mean())),
                        "mean_d_se": float(dd.mean()), "n_s_better": int((dd > 0).sum()),
                        "p_ttest": t_p, "p_wilcoxon": w_p})
    df = pd.DataFrame(out)
    order = {k: i for i, k in enumerate([*SNAPSHOTS, "ALL"])}
    return df.sort_values(["base", "snapshot"], key=lambda c: c.map(order) if c.name == "snapshot"
                          else c).reset_index(drop=True)

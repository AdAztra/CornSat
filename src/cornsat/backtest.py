"""Phase 6, Stage B: weekly ZCZ trading backtest of the yield-surprise model.

Conventions (see reports/phase6_design.md)
-------------------------------------------
* Instrument: one ZCZ contract of the campaign year (5,000 bu, so 1 cent = $50). There are
  no rolls in Jun-Sep.
* Rebalance on every Tuesday from the first on or after Jun 1 to Sep 30; if Tuesday is not
  a trading day, use the next trading day of that week (``rebalance_dates``). Trades are
  at that day's close. The position is closed at the close of the last trading day <=
  Sep 30 (``exit_date``) and is flat outside the window. A rebalance falling on the exit
  day is dropped, since the position must be flat there anyway.
* Costs: COST_PER_SIDE = $2.50 (half of the $5 round-trip commission) + $12.50 (0.25 cent
  slippage) = $15 per contract per side, charged as |delta position| x $15 at every change.
* Weekly period k runs from execution date k to the next execution date, or to the exit.
  Its net P&L is pos_k x (close_end - close_start) x $50, minus the cost of trading into
  pos_k, minus the exit cost for the season's last period.
* Sharpe: weekly net P&L / notional, where notional is the close at the season's first
  execution x $50. Then mean / sd x sqrt(52) over all in-window weeks, flat weeks included.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from cornsat.config import CONTRACT_BUSHELS, TRADE_END, TRADE_START
from cornsat.model import ALPHAS, MIN_TRAIN, rebalance_predictions, usda_surprise_at

DOLLARS_PER_CENT = CONTRACT_BUSHELS / 100.0  # 50
COMMISSION_RT = 5.0
SLIPPAGE_CENTS = 0.25
COST_PER_SIDE = COMMISSION_RT / 2 + SLIPPAGE_CENTS * DOLLARS_PER_CENT  # 15
WEEKS_PER_YEAR = 52

TRADE_SETS = ["W", "C", "W+C", "W+C+S", "S"]
SIGNALS = ["level", "change", "vs_usda"]
THRESHOLDS = {"level": 2.0, "change": 0.5, "vs_usda": 2.0}
BENCHMARKS = ["always_short", "always_long", "flat"]


# --- calendar ----------------------------------------------------------------------------
def season_prices(prices: pd.DataFrame, season: int) -> pd.Series:
    """Close (cents) of the front-December contract of ``season`` on dates in that year."""
    p = prices[(prices["campaign"] == season) & (pd.to_datetime(prices["date"]).dt.year == season)]
    s = pd.Series(p["close"].to_numpy(float), index=pd.to_datetime(p["date"]), name="close")
    return s.sort_index()


def exit_date(trading_dates: pd.DatetimeIndex, season: int, end=TRADE_END) -> pd.Timestamp:
    """Last trading date <= end (month, day) of ``season``."""
    d = trading_dates[trading_dates <= pd.Timestamp(season, *end)]
    return d.max()


def rebalance_dates(trading_dates: pd.DatetimeIndex, season: int, start=TRADE_START,
                    end=TRADE_END) -> list[pd.Timestamp]:
    """Tuesdays from the first on or after ``start`` to ``end``. A Tuesday that is not a
    trading day moves to the next trading day of the same week. Dates on or after the
    exit date are dropped."""
    td = pd.DatetimeIndex(sorted(trading_dates))
    ex = exit_date(td, season, end)
    lo, hi = pd.Timestamp(season, *start), pd.Timestamp(season, *end)
    tues = pd.date_range(lo, hi, freq="W-TUE")
    out = []
    for t in tues:
        c = td[(td >= t) & (td < t + pd.Timedelta(days=5))]  # Tue .. Sat
        if len(c) and c[0] < ex:
            out.append(c[0])
    return sorted(set(out))


def next_trading_day(trading_dates: pd.DatetimeIndex, d: pd.Timestamp):
    td = trading_dates[trading_dates > d]
    return td.min() if len(td) else pd.NaT


# --- P&L ---------------------------------------------------------------------------------
def simulate_season(close: pd.Series, exec_dates, positions, exit_dt,
                    cost_per_side: float = COST_PER_SIDE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily and weekly P&L for one season.

    ``close``: close in cents indexed by trading date. ``exec_dates``/``positions``: target
    position set at the close of each execution date. ``exit_dt``: last day; the position is
    0 at its close. Execution dates on or after ``exit_dt`` are ignored.
    Returns (daily, weekly). daily covers first execution .. exit with columns pos (held
    after the close), gross, cost, net in $. weekly has one row per holding period."""
    close = close.sort_index()
    ex = pd.DatetimeIndex(pd.to_datetime(list(exec_dates)))
    pos = np.asarray(list(positions), dtype=float)
    keep = ex < exit_dt
    ex, pos = ex[keep], pos[keep]
    if len(ex) == 0:
        return (pd.DataFrame(columns=["pos", "gross", "cost", "net"]),
                pd.DataFrame(columns=["start", "end", "pos", "d_close", "gross", "cost", "net"]))
    days = close.index[(close.index >= ex[0]) & (close.index <= exit_dt)]
    if not set(ex).issubset(set(days)):
        raise ValueError("execution dates must be trading dates")
    c = close.loc[days].to_numpy(float)
    if np.isnan(c).any():
        raise ValueError("NaN close inside the trading window")
    k = np.searchsorted(ex.to_numpy(), days.to_numpy(), side="right") - 1
    held = pos[k]
    held[-1] = 0.0  # flat at the exit close
    prev = np.concatenate([[0.0], held[:-1]])
    gross = np.concatenate([[0.0], prev[1:] * np.diff(c) * DOLLARS_PER_CENT])
    cost = np.abs(held - prev) * cost_per_side
    daily = pd.DataFrame({"pos": held, "gross": gross, "cost": cost, "net": gross - cost},
                         index=days)
    ends = [*ex[1:], days[-1]]
    weekly = []
    for i, (s, e) in enumerate(zip(ex, ends)):
        inside = (days > s) & (days <= e)
        g = float(gross[inside].sum())
        cst = float(cost[days == s].sum())
        if i == len(ex) - 1:
            cst += float(cost[-1])
        weekly.append({"start": s, "end": e, "pos": float(pos[i]),
                       "d_close": float(close.loc[e] - close.loc[s]),
                       "gross": g, "cost": cst, "net": g - cst})
    return daily, pd.DataFrame(weekly)


def simulate(prices: pd.DataFrame, schedule: pd.DataFrame, exec_lag: int = 0,
             cost_per_side: float = COST_PER_SIDE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run ``schedule`` (columns season, date, position; one row per rebalance) through
    the P&L. ``exec_lag`` = 1 trades at the next trading day's close instead of on the
    rebalance date. Returns concatenated (daily, weekly) with a ``season`` column and
    weekly ``notional`` (close at the season's first execution x $50)."""
    dailies, weeklies = [], []
    for y, g in schedule.groupby("season"):
        close = season_prices(prices, int(y))
        td = close.index
        ex_dt = exit_date(td, int(y))
        dates = pd.to_datetime(g["date"]).tolist()
        if exec_lag:
            dates = [next_trading_day(td, d) for d in dates]
        ok = [pd.notna(d) and d < ex_dt for d in dates]
        dates = [d for d, o in zip(dates, ok) if o]
        posn = g["position"].to_numpy(float)[np.asarray(ok, bool)]
        d, w = simulate_season(close, dates, posn, ex_dt, cost_per_side)
        if len(w):
            w["notional"] = float(close.loc[w["start"].iloc[0]]) * DOLLARS_PER_CENT
        dailies.append(d.assign(season=int(y)))
        weeklies.append(w.assign(season=int(y)))
    daily = pd.concat(dailies).rename_axis("date").reset_index()
    weekly = pd.concat(weeklies, ignore_index=True)
    return daily, weekly


# --- signals -----------------------------------------------------------------------------
def level_position(pred, thr: float) -> np.ndarray:
    """-sign(pred) where |pred| > thr; 0 otherwise or if pred is NaN."""
    p = np.asarray(pred, dtype=float)
    return np.where(np.abs(np.nan_to_num(p)) > thr, -np.sign(np.nan_to_num(p)), 0.0)


def change_position(pred: pd.Series, season: pd.Series, thr: float) -> np.ndarray:
    """-sign(delta pred vs the previous rebalance of the same season) where |delta| > thr.
    The first rebalance of a season, and NaN predictions, give 0."""
    delta = pd.Series(np.asarray(pred, float)).groupby(np.asarray(season)).diff()
    return level_position(delta.to_numpy(), thr)


def vs_usda_position(pred, usda, thr: float) -> np.ndarray:
    """-sign(pred - USDA surprise) where |gap| > thr; 0 before the first forecast."""
    gap = np.asarray(pred, float) - np.asarray(usda, float)
    return level_position(gap, thr)


def build_schedule(trading_dates_by_season: dict[int, pd.DatetimeIndex]) -> pd.DataFrame:
    """(season, date) of every rebalance."""
    rows = [(y, d) for y, td in sorted(trading_dates_by_season.items())
            for d in rebalance_dates(td, y)]
    return pd.DataFrame(rows, columns=["season", "date"])


def compute_signals(feat: pd.DataFrame, targets: pd.DataFrame, forecasts: pd.DataFrame,
                    schedule: pd.DataFrame, sets=TRADE_SETS, alphas=ALPHAS,
                    min_train: int = MIN_TRAIN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(wide, long). wide: one row per rebalance with the walk-forward prediction per set
    (``pred_<set>``), flags (``fallback_<set>``, ``sdrop_<set>``) and ``usda`` (latest
    USDA forecast surprise usable at the rebalance date). long: ``rebalance_predictions``."""
    preds = rebalance_predictions(feat, targets, schedule, sets, alphas, min_train)
    wide = schedule.copy()
    wide["date"] = pd.to_datetime(wide["date"])
    for s in sets:
        p = preds[preds["set"] == s].set_index(["season", "date"])
        key = pd.MultiIndex.from_frame(wide[["season", "date"]])
        wide[f"pred_{s}"] = p["pred"].reindex(key).to_numpy()
        wide[f"fallback_{s}"] = p["fallback"].reindex(key).to_numpy()
        wide[f"sdrop_{s}"] = p["s_dropped"].reindex(key).to_numpy()
    wide["usda"] = [usda_surprise_at(forecasts, targets, y, d)
                    for y, d in zip(wide["season"], wide["date"])]
    return wide, preds


def positions_table(signals: pd.DataFrame, sets=TRADE_SETS, thresholds=None) -> pd.DataFrame:
    """Position per strategy ("<signal>|<set>" and benchmarks) at every rebalance."""
    thr = {**THRESHOLDS, **(thresholds or {})}
    out = signals[["season", "date"]].copy()
    for s in sets:
        p = signals[f"pred_{s}"]
        out[f"level|{s}"] = level_position(p, thr["level"])
        out[f"change|{s}"] = change_position(p, signals["season"], thr["change"])
        out[f"vs_usda|{s}"] = vs_usda_position(p, signals["usda"], thr["vs_usda"])
    out["always_short"] = -1.0
    out["always_long"] = 1.0
    out["flat"] = 0.0
    return out


def compute_positions(feat, targets, forecasts, schedule, sets=TRADE_SETS, thresholds=None,
                      alphas=ALPHAS, min_train: int = MIN_TRAIN) -> pd.DataFrame:
    """Signals and positions in one step (used by the no-look-ahead test)."""
    sig, _ = compute_signals(feat, targets, forecasts, schedule, sets, alphas, min_train)
    return positions_table(sig, sets, thresholds)


def strategy_names(positions: pd.DataFrame) -> list[str]:
    return [c for c in positions.columns if c not in ("season", "date")]


def run_strategies(prices: pd.DataFrame, positions: pd.DataFrame, exec_lag: int = 0,
                   names: Iterable[str] | None = None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """{strategy: (daily, weekly)} for every strategy column of ``positions``."""
    names = list(names) if names is not None else strategy_names(positions)
    out = {}
    for n in names:
        sch = positions[["season", "date"]].assign(position=positions[n].to_numpy(float))
        out[n] = simulate(prices, sch, exec_lag=exec_lag)
    return out


# --- metrics -----------------------------------------------------------------------------
def sharpe(weekly: pd.DataFrame) -> float:
    r = weekly["net"] / weekly["notional"]
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(WEEKS_PER_YEAR)) if sd > 0 else np.nan


def max_drawdown(daily: pd.DataFrame) -> float:
    eq = daily.sort_values("date")["net"].cumsum().to_numpy()
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]
    return float((peak - eq).max())


def summarize(daily: pd.DataFrame, weekly: pd.DataFrame) -> dict:
    act = weekly[weekly["pos"] != 0]
    dpos = daily.groupby("season")["pos"].apply(
        lambda s: np.abs(np.diff(np.concatenate([[0.0], s.to_numpy()]))))
    changes = int(sum((d > 0).sum() for d in dpos))
    season_pnl = weekly.groupby("season")["net"].sum()
    return {
        "net_pnl": float(weekly["net"].sum()), "gross_pnl": float(weekly["gross"].sum()),
        "costs": float(weekly["cost"].sum()), "sharpe": sharpe(weekly),
        "hit_rate": float((act["gross"] > 0).mean()) if len(act) else np.nan,
        "active_weeks": int(len(act)), "weeks": int(len(weekly)),
        "max_dd": max_drawdown(daily), "trades": changes,
        "contracts": float(sum(d.sum() for d in dpos)),
        "seasons_pos": int((season_pnl > 0).sum()), "seasons": int(len(season_pnl)),
        "pnl_per_season": float(season_pnl.mean()),
    }


def summary_table(results: dict) -> pd.DataFrame:
    rows = [{"strategy": n, **summarize(d, w)} for n, (d, w) in results.items()]
    return pd.DataFrame(rows).set_index("strategy")


def season_table(results: dict) -> pd.DataFrame:
    return pd.DataFrame({n: w.groupby("season")["net"].sum() for n, (d, w) in results.items()})


# --- robustness --------------------------------------------------------------------------
def block_bootstrap(weekly: pd.DataFrame, n_boot: int = 10_000, seed: int = 0,
                    other: pd.DataFrame | None = None) -> dict:
    """Season-block bootstrap of total net P&L and Sharpe (seasons resampled with
    replacement). With ``other``, also the paired difference in total P&L (this - other)."""
    rng = np.random.default_rng(seed)
    seasons = np.array(sorted(weekly["season"].unique()))
    r = weekly["net"] / weekly["notional"]
    by_r = {s: r[weekly["season"] == s].to_numpy() for s in seasons}
    tot = weekly.groupby("season")["net"].sum().reindex(seasons).to_numpy()
    oth = (other.groupby("season")["net"].sum().reindex(seasons).to_numpy()
           if other is not None else None)
    draws = rng.integers(0, len(seasons), size=(n_boot, len(seasons)))
    totals = tot[draws].sum(axis=1)
    sh = np.empty(n_boot)
    for b in range(n_boot):
        x = np.concatenate([by_r[seasons[i]] for i in draws[b]])
        sd = x.std(ddof=1)
        sh[b] = x.mean() / sd * np.sqrt(WEEKS_PER_YEAR) if sd > 0 else np.nan
    out = {"pnl_lo": np.percentile(totals, 2.5), "pnl_hi": np.percentile(totals, 97.5),
           "p_pnl_le0": float((totals <= 0).mean()),
           "sharpe_lo": np.nanpercentile(sh, 2.5), "sharpe_hi": np.nanpercentile(sh, 97.5)}
    if oth is not None:
        diff = (tot - oth)[draws].sum(axis=1)
        out.update(diff=float((tot - oth).sum()), diff_lo=np.percentile(diff, 2.5),
                   diff_hi=np.percentile(diff, 97.5), p_diff_le0=float((diff <= 0).mean()))
    return out


def _random_totals(weekly: pd.DataFrame, abs_pos: np.ndarray, n: int, rng,
                   cost_per_side: float = COST_PER_SIDE) -> np.ndarray:
    """Total net P&L of ``n`` random-sign versions of |position| ``abs_pos`` (one per
    weekly period), costs recomputed per season (entry from 0, exit to 0)."""
    tot = np.zeros(n)
    for s, g in weekly.groupby("season", sort=True):
        idx = g.index.to_numpy()
        a = abs_pos[weekly.index.get_indexer(idx)]
        dp = g["d_close"].to_numpy(float) * DOLLARS_PER_CENT
        sign = rng.choice([-1.0, 1.0], size=(n, len(idx)))
        p = sign * a
        padded = np.concatenate([np.zeros((n, 1)), p, np.zeros((n, 1))], axis=1)
        cost = np.abs(np.diff(padded, axis=1)).sum(axis=1) * cost_per_side
        tot += p @ dp - cost
    return tot


def random_sign_tests(weekly: pd.DataFrame, n: int = 5_000, seed: int = 1) -> dict:
    """(a) iid +/-1 every week; (b) random signs on the strategy's own active weeks.
    p = share of random draws with total net P&L >= the strategy's (with +1 smoothing)."""
    rng = np.random.default_rng(seed)
    actual = float(weekly["net"].sum())
    w = weekly.reset_index(drop=True)
    full = _random_totals(w, np.ones(len(w)), n, rng)
    own = _random_totals(w, np.abs(w["pos"].to_numpy(float)), n, rng)
    return {"p_random_full": (1 + (full >= actual).sum()) / (n + 1),
            "p_random_own": (1 + (own >= actual).sum()) / (n + 1),
            "random_full_median": float(np.median(full)),
            "random_full_p95": float(np.percentile(full, 95))}

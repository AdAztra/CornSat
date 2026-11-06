import pandas as pd
import pytest

from cornsat.config import DATA_RAW, YEARS
from cornsat.prices import expiry_dates, front_dec_series, load_all

TOL = 0.5  # cents; real data may have tiny violations


@pytest.fixture(scope="module")
def prices():
    return load_all(DATA_RAW)


@pytest.fixture(scope="module")
def front(prices):
    return front_dec_series(prices)


def test_all_contracts_load(prices):
    assert prices["contract"].nunique() == 18
    assert sorted(prices["campaign"].unique()) == list(YEARS)


def test_no_footer_and_dtypes(prices):
    assert pd.api.types.is_datetime64_any_dtype(prices["date"])
    assert prices["date"].notna().all()
    assert prices["close"].dtype == float
    assert prices["volume"].dtype.kind == "i"


def test_dates_unique_ascending(prices):
    for c, g in prices.groupby("contract"):
        assert g["date"].is_unique, c
        assert g["date"].is_monotonic_increasing, c


def test_ohlc_consistency(prices):
    p = prices
    bad = p[
        (p.close > p.high + TOL) | (p.close < p.low - TOL)
        | (p.open > p.high + TOL) | (p.open < p.low - TOL)
    ]
    assert bad.empty, bad[["contract", "date", "open", "high", "low", "close"]].to_string()


def test_expiry(prices):
    exp = expiry_dates(prices)
    assert pd.isna(exp["ZCZ26"])
    assert exp.drop("ZCZ26").notna().all()
    assert (exp.dropna().dt.month == 12).all()


def test_no_roll_in_summer(front):
    changed = front["contract"] != front["contract"].shift(1)
    rolls = front[changed].iloc[1:]
    assert len(rolls) == 17
    assert not rolls["date"].dt.month.between(6, 9).any()


def test_campaigns_present(front):
    assert set(front["campaign"]) == set(YEARS)
    assert front["date"].min() == pd.Timestamp("2009-01-02")


def test_ret_nan_exactly_on_roll_days(front):
    roll = front["contract"] != front["contract"].shift(1)
    assert front["ret"].isna().equals(roll)


def test_known_value(prices):
    row = prices[(prices.contract == "ZCZ09") & (prices.date == "2009-12-11")]
    assert row["close"].iloc[0] == 389.25

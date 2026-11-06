import pandas as pd
import pytest

from cornsat.config import DATA_RAW, YEARS
from cornsat.prices import download_date, expiry_dates, front_dec_series, load_all, read_contract

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


def _write_csv(path, rows, footer):
    lines = ["Time,Open,High,Low,Latest,Change,%Change,Volume,\"Open Int\""]
    lines += rows + [footer]
    path.write_text("\n".join(lines) + "\n")


def test_partial_bar_on_download_date_dropped(tmp_path):
    rows = ["2026-11-05,445.5,447.75,444.75,445.75,1.5,+0.34%,17595,0",
            "2026-11-04,441.75,447.25,438.5,444.25,1.75,+0.40%,212688,734618",
            "2026-11-03,440,445,438,442.5,0.5,+0.11%,1000,2000"]
    footer = '"Downloaded from Barchart.com as of 11-05-2026 07:10am CST"'
    f = tmp_path / "zcz26_daily_historical-data-11-05-2026.csv"
    _write_csv(f, rows, footer)
    assert download_date(f) == pd.Timestamp("2026-11-05")
    df = read_contract(f)
    assert list(df["date"]) == list(pd.to_datetime(["2026-11-03", "2026-11-04"]))
    # file name without a date: fall back to the footer
    g = tmp_path / "zcz26_daily_historical-data.csv"
    _write_csv(g, rows, footer)
    assert download_date(g, footer) == pd.Timestamp("2026-11-05")
    assert read_contract(g)["date"].max() == pd.Timestamp("2026-11-04")


def test_real_data_has_no_row_on_download_date(prices):
    assert prices["date"].max() < pd.Timestamp("2026-11-05")
    last = prices[prices["contract"] == "ZCZ26"]["date"].max()
    assert last == pd.Timestamp("2026-11-04")

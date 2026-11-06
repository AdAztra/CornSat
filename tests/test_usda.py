import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cornsat import usda

FX = Path(__file__).parent / "fixtures"


def load(name):
    with open(FX / name) as f:
        return pd.DataFrame(json.load(f)["data"])


def test_parse_value():
    assert usda.parse_value("1,234") == 1234
    assert usda.parse_value("123.1") == 123.1
    for bad in ["(D)", "(NA)", "", None]:
        assert np.isnan(usda.parse_value(bad))


def test_condition_index():
    assert usda.condition_index(0, 0, 0, 0, 100) == 5
    assert usda.condition_index(20, 20, 20, 20, 20) == pytest.approx(3.0)


def test_release_dates():
    # normal week: Sun 2024-06-02 -> Tue 06-04
    assert usda.crop_progress_available_date("2024-06-02") == pd.Timestamp("2024-06-04")
    # Labor Day 2024-09-02 (Mon): week ending 09-01 -> Wed 09-04
    assert usda.crop_progress_available_date("2024-09-01") == pd.Timestamp("2024-09-04")
    # Memorial Day 2024-05-27: week ending 05-26 -> Wed 05-29
    assert usda.crop_progress_available_date("2024-05-26") == pd.Timestamp("2024-05-29")


def test_production_release_dates():
    # Aug 10 2024 is Saturday -> release Mon 12th -> usable Tue 13th
    assert usda.crop_production_release_date(2024, 8) == pd.Timestamp("2024-08-13")
    # Jan 10 2025 is Friday -> usable Mon 13th
    assert usda.crop_production_release_date(2025, 1) == pd.Timestamp("2025-01-13")
    assert usda.final_yield_available_date(2024) == pd.Timestamp("2025-01-13")


def test_condition_fixture():
    c = usda.parse_condition(load("usda_condition.json"))
    assert set(c.geo) == {"US", "IA"}  # incomplete (NA) week dropped
    us = c[c.geo == "US"].iloc[0]
    assert us["ge"] == 27 and us["ci"] == pytest.approx((22 + 50 + 78 + 84 + 30) / 100)
    assert us.available_date == pd.Timestamp("2012-07-31")


def test_progress_fixture():
    p = usda.parse_progress(load("usda_progress.json"))
    assert len(p) == 1 and p.pct_planted[0] == 100 and np.isnan(p.pct_mature[0])
    assert p.available_date[0] == pd.Timestamp("2012-09-05")  # Labor Day 2012-09-03


def test_yield_fixture(monkeypatch):
    monkeypatch.setattr(usda, "today", lambda: pd.Timestamp("2026-11-06"))
    y = usda.parse_yield(load("usda_yield.json"))
    assert set(y.period) == {"AUG", "SEP", "FINAL"}  # INTENTIONS row ignored
    f = y[y.period == "FINAL"].iloc[0]
    assert f.yield_bu_ac == 123.1 and f.production_bu == 10780296000
    assert f.available_date == usda.final_yield_available_date(2012)
    # AUG load_time Fri 2012-08-10 -> Mon 08-13
    assert y[y.period == "AUG"].iloc[0].available_date == pd.Timestamp("2012-08-13")
    # SEP load_time Tue 2012-09-11 -> Wed 09-12
    assert y[y.period == "SEP"].iloc[0].available_date == pd.Timestamp("2012-09-12")


def test_next_business_day_after():
    assert usda.next_business_day_after("2012-08-10 12:59:15") == pd.Timestamp("2012-08-13")  # Fri
    # Fri 2024-05-24 load, Memorial Day Mon 05-27 -> Tue 05-28
    assert usda.next_business_day_after("2024-05-24") == pd.Timestamp("2024-05-28")


def test_forecast_load_time_cases():
    # 2026 AUG: rule says 08-11 (look-ahead); load_time 08-12 -> Thu 08-13
    assert usda.crop_production_release_date(2026, 8) == pd.Timestamp("2026-08-11")
    assert usda.forecast_available_date(2026, 8, ["2026-08-12 12:00:00.000"]) == pd.Timestamp("2026-08-13")
    # holiday: load Tue 2020-11-10, Veterans Day Wed 11-11 -> Thu 11-12
    assert usda.forecast_available_date(2020, 11, ["2020-11-10 12:00:00"]) == pd.Timestamp("2020-11-12")
    # max over several load_times
    assert usda.forecast_available_date(
        2012, 8, ["2012-08-10 12:59:15", "2012-08-14 09:00:00"]) == pd.Timestamp("2012-08-15")
    # fallback: missing, unparsable, or wrong month
    rule = usda.crop_production_release_date(2012, 8)
    for lt in (None, [None], ["garbage"], ["2012-09-10 12:00:00"]):
        assert usda.forecast_available_date(2012, 8, lt) == rule


def test_yield_rule_fallback_without_load_time(monkeypatch):
    monkeypatch.setattr(usda, "today", lambda: pd.Timestamp("2026-11-06"))
    raw = load("usda_yield.json").drop(columns=["load_time"])
    y = usda.parse_yield(raw)
    assert y[y.period == "AUG"].iloc[0].available_date == usda.crop_production_release_date(2012, 8)


def test_final_future_dropped(monkeypatch):
    raw = load("usda_yield.json")
    monkeypatch.setattr(usda, "today", lambda: pd.Timestamp("2013-01-10"))  # day before 2013-01-11
    assert usda.final_yield_available_date(2012) == pd.Timestamp("2013-01-11")
    assert "FINAL" not in set(usda.parse_yield(raw).period)
    monkeypatch.setattr(usda, "today", lambda: pd.Timestamp("2013-01-11"))
    assert "FINAL" in set(usda.parse_yield(raw).period)
    # current calendar year is always dropped
    monkeypatch.setattr(usda, "today", lambda: pd.Timestamp("2012-12-31"))
    assert "FINAL" not in set(usda.parse_yield(raw).period)


def test_missing_key(monkeypatch):
    monkeypatch.setattr(usda, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("NASS_API_KEY", raising=False)
    with pytest.raises(usda.NassKeyMissing, match="quickstats.nass.usda.gov/api"):
        usda.get_api_key()

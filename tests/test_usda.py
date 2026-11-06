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


def test_yield_fixture():
    y = usda.parse_yield(load("usda_yield.json"))
    assert set(y.period) == {"AUG", "SEP", "FINAL"}  # INTENTIONS row ignored
    f = y[y.period == "FINAL"].iloc[0]
    assert f.yield_bu_ac == 123.1 and f.production_bu == 10780296000
    assert f.available_date == usda.final_yield_available_date(2012)
    assert y[y.period == "AUG"].iloc[0].available_date == usda.crop_production_release_date(2012, 8)


def test_missing_key(monkeypatch):
    monkeypatch.setattr(usda, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("NASS_API_KEY", raising=False)
    with pytest.raises(usda.NassKeyMissing, match="quickstats.nass.usda.gov/api"):
        usda.get_api_key()

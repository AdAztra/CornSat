"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"

YEARS = range(2009, 2027)

# State abbreviation -> FIPS code
STATES = {"IA": "19", "IL": "17", "NE": "31", "MN": "27", "IN": "18"}
EXTRA_STATES = {"SD": "46", "OH": "39", "WI": "55", "MO": "29", "KS": "20"}

# (month, day) tuples
SEASON_START = (4, 1)
SEASON_END = (10, 31)
TRADE_START = (6, 1)
TRADE_END = (9, 30)

CONTRACT_BUSHELS = 5000

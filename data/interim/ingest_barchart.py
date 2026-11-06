from pathlib import Path

import pandas as pd


def read_contract(path: Path, campaing: int) -> pd.DataFrame:
    """Reads a CSV from Barchart and returns normalized columns"""
    raw = pd.read_csv(path, dtype="str")
    is_footer = raw["Time"].str.startswith("Downloaded", na=False)

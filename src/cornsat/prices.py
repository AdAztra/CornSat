"""Ingest Barchart ZCZ (December corn) daily exports."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from cornsat.config import DATA_PROCESSED, DATA_RAW

FRONT_START = pd.Timestamp("2009-01-02")


def download_date(path: Path, footer: str | None = None) -> pd.Timestamp | None:
    """Download date of a Barchart export: the ``MM-DD-YYYY`` in the file name
    (``..._historical-data-11-05-2026.csv``), else the ``as of MM-DD-YYYY`` of the footer
    line. None if neither parses."""
    m = re.search(r"(\d{2})-(\d{2})-(\d{4})\.csv$", Path(path).name.lower())
    if not m and footer:
        m = re.search(r"as of (\d{2})-(\d{2})-(\d{4})", footer)
    if not m:
        return None
    return pd.Timestamp(int(m.group(3)), int(m.group(1)), int(m.group(2)))


def read_contract(path: Path) -> pd.DataFrame:
    """Read one Barchart CSV and return normalized, ascending columns.

    A row dated on the file's download date is dropped: the export was taken during/before
    that session, so the bar is a partial intraday snapshot, not a settled close."""
    path = Path(path)
    m = re.search(r"zcz(\d{2})", path.name.lower())
    if not m:
        raise ValueError(f"cannot parse campaign year from {path.name}")
    campaign = 2000 + int(m.group(1))
    raw = pd.read_csv(path, dtype=str)
    footer = raw["Time"][raw["Time"].str.startswith("Downloaded", na=False)]
    dl = download_date(path, footer.iloc[-1] if len(footer) else None)
    raw = raw[~raw["Time"].str.startswith("Downloaded", na=False)].copy()
    raw = raw.dropna(subset=["Time"])
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(raw["Time"]),
            "open": raw["Open"].astype(float),
            "high": raw["High"].astype(float),
            "low": raw["Low"].astype(float),
            "close": raw["Latest"].astype(float),
            "change": raw["Change"].astype(float),
            "pct_change": raw["%Change"].str.replace("%", "").astype(float) / 100,
            "volume": raw["Volume"].astype("int64"),
            "oi": raw["Open Int"].astype("int64"),
        }
    )
    if dl is not None:
        df = df[df["date"] < dl]
    df = df.drop_duplicates().drop_duplicates(subset="date", keep="first")
    df = df.sort_values("date").reset_index(drop=True)
    df["contract"] = f"ZCZ{campaign % 100:02d}"
    df["campaign"] = campaign
    return df


def load_all(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    files = sorted(Path(raw_dir).glob("zcz??_daily_historical-data-*.csv"))
    if not files:
        raise FileNotFoundError(f"no ZCZ csv files in {raw_dir}")
    return pd.concat([read_contract(f) for f in files], ignore_index=True)


def expiry_dates(prices: pd.DataFrame) -> pd.Series:
    """Last trading date per contract; NaT if not yet expired (last date not in Dec of its campaign)."""
    g = prices.groupby("contract").agg(last=("date", "max"), campaign=("campaign", "first"))
    expired = (g["last"].dt.month == 12) & (g["last"].dt.year == g["campaign"])
    return g["last"].where(expired).rename("expiry")


def front_dec_series(prices: pd.DataFrame) -> pd.DataFrame:
    """Front December contract per date, rolling right after each contract's expiry."""
    exp = expiry_dates(prices)
    by_c = {c: set(d["date"]) for c, d in prices.groupby("contract")}
    dates = sorted(prices.loc[prices["date"] >= FRONT_START, "date"].unique())
    rows = []
    for d in dates:
        d = pd.Timestamp(d)
        y = d.year
        e = exp.get(f"ZCZ{y % 100:02d}", pd.NaT)
        if pd.notna(e) and d > e:
            y += 1
        c = f"ZCZ{y % 100:02d}"
        if c in by_c and d in by_c[c]:
            rows.append((d, c, y))
    sel = pd.DataFrame(rows, columns=["date", "contract", "campaign"])
    cols = ["open", "high", "low", "close", "volume", "oi"]
    long = prices.set_index(["contract", "date"])[cols]
    sel = sel.join(long, on=["contract", "date"])
    e_map = sel["contract"].map(exp)
    sel["days_to_expiry"] = (e_map - sel["date"]).dt.days
    prev = sel.groupby("contract")["close"].shift(1)
    sel["ret"] = np.log(sel["close"] / prev)
    return sel[["date", "contract", "campaign", *cols, "days_to_expiry", "ret"]]


def save_processed(raw_dir: Path = DATA_RAW, out_dir: Path = DATA_PROCESSED):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = load_all(raw_dir)
    front = front_dec_series(prices)
    prices.to_parquet(out_dir / "zcz_prices.parquet", index=False)
    front.to_parquet(out_dir / "zcz_front_dec.parquet", index=False)
    return prices, front

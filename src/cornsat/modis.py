"""Corn-weighted MODIS vegetation indices (NDVI, EVI) per state and 16-day composite.

Source
------
Microsoft Planetary Computer STAC, collection ``modis-13Q1-061`` (MODIS Collection 6.1
MOD13Q1 = Terra and MYD13Q1 = Aqua, 250 m, 16-day composites; the two platforms are
offset by 8 days, so together they give an 8-day cadence). Item ids look like
``MOD13Q1.A2012193.h10v04.061.<production>``; ``properties["platform"]`` is ``terra`` or
``aqua``; items have ``datetime = null`` and a 16-day ``start_datetime``/``end_datetime``
window (e.g. 2012-07-11T00:00:00 .. 2012-07-26T23:59:59).

Assets used (one COG per band, 4800x4800 px per sinusoidal tile, 512 px blocks):

=================================== ======== ======= =========================================
asset                               dtype    fill    meaning
=================================== ======== ======= =========================================
250m_16_days_NDVI                   int16    -3000   NDVI x 10000, valid range -2000..10000
250m_16_days_EVI                    int16    -3000   EVI x 10000, valid range -2000..10000
250m_16_days_pixel_reliability      uint8    255     0 good, 1 marginal, 2 snow/ice, 3 cloudy
250m_16_days_composite_day_of_the   int16    -1      day of year the chosen pixel was observed
  _year
=================================== ======== ======= =========================================

STAC ``raster:bands`` gives ``scale = 0.0001`` for NDVI/EVI and labels pixel_reliability
``int8`` (where the product's fill is -1); the COG on disk is actually ``uint8`` with
fill 255 (= -1 as int8). ``apply_qa`` keeps only pixels whose reliability is in
``keep`` so either encoding of the fill is rejected.

Grid
----
Composites are loaded with ``odc.stac.load`` straight onto ``cornmask.modis_geobox``
(native MODIS grid, integer pixel offsets into the tiles), so there is no resampling;
the 2-4 tiles covering a state are mosaicked (tiles abut exactly, no overlap).

Weights
-------
Each pixel is weighted by ``cornmask.season_weight(state, year)``: the corn fraction from
CDL years strictly before the season (no look-ahead), zero outside the state.
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

from cornsat.config import DATA_INTERIM, DATA_PROCESSED, SEASON_END, SEASON_START, STATES
from cornsat.cornmask import modis_geobox, season_weight

log = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "modis-13Q1-061"
MODIS_DIR = DATA_INTERIM / "modis"
OUT_PATH = DATA_PROCESSED / "modis_state.parquet"

# asset name -> (short name, nodata on disk, dtype on disk)
BANDS: dict[str, tuple[str, int, str]] = {
    "250m_16_days_NDVI": ("ndvi", -3000, "int16"),
    "250m_16_days_EVI": ("evi", -3000, "int16"),
    "250m_16_days_pixel_reliability": ("reliability", 255, "uint8"),
    "250m_16_days_composite_day_of_the_year": ("doy", -1, "int16"),
}
SCALE = 0.0001
FILL = -3000
VALID_MIN, VALID_MAX = -0.2, 1.0
RELIABILITY = {0: "good", 1: "marginal", 2: "snow/ice", 3: "cloudy", 255: "fill (-1 as int8)"}
PLATFORMS = {"MOD13Q1": "terra", "MYD13Q1": "aqua"}

LAG_DAYS = 10
"""Days after a composite's last day before its values may be used (``available_date``).

MOD13Q1/MYD13Q1 are produced after the 16-day window closes and then published to the
archives; routine (non-NRT) production has typically landed within roughly a week of
``end_date``. 10 days is a deliberately conservative production + publication lag. The
backtest must only use a composite on or after ``available_date`` (no look-ahead).
Caveat: the values themselves are Collection 6.1 reprocessed in 2021, not the exact
numbers that were downloadable in real time; the lag models timing only.
"""

COLUMNS = [
    "state", "year", "platform", "start_date", "end_date", "mid_date", "available_date",
    "obs_date", "ndvi_mean", "evi_mean", "ndvi_mean_good", "evi_mean_good", "ndvi_median",
    "valid_frac", "valid_frac_good", "n_tiles", "cdl_years",
]  # fmt: skip

_RETRIES = 5


# --- pure functions -----------------------------------------------------------------
def apply_qa(ndvi_raw, evi_raw, reliability, keep=(0, 1)):
    """Mask and scale raw MOD13Q1 NDVI/EVI.

    A pixel is kept when its pixel_reliability is in ``keep`` (so -1/255 fill, 2 snow/ice
    and 3 cloudy are dropped by default), the raw value is not the -3000 fill, and the
    scaled value lies in [VALID_MIN, VALID_MAX] = [-0.2, 1.0]. Returns float32
    ``(ndvi, evi)`` with NaN where masked. Works on numpy arrays or xarray DataArrays.
    """
    rel = np.asarray(reliability)
    ok_rel = np.isin(rel, np.asarray(keep, dtype=rel.dtype)) if len(keep) else np.zeros(rel.shape, bool)

    def one(raw):
        r = np.asarray(raw)
        v = r.astype("float32") * np.float32(SCALE)
        bad = ~ok_rel | (r == FILL) | ~np.isfinite(v) | (v < VALID_MIN) | (v > VALID_MAX)
        v[bad] = np.nan
        if isinstance(raw, xr.DataArray):
            return raw.copy(data=v)
        return v

    return one(ndvi_raw), one(evi_raw)


def weighted_stats(values, weights) -> dict[str, float]:
    """Weighted mean of ``values`` (NaN = no valid data) with ``weights`` (corn fraction).

    Returns ``mean`` = sum(w*v)/sum(w) over valid pixels, ``valid_frac`` = share of the
    total weight that had valid data (so cloud-masked corn is visible), and ``weight``
    (total weight). Non-finite or negative weights count as 0. All-NaN values give
    mean NaN and valid_frac 0; zero total weight gives NaN for both.
    """
    v = np.asarray(values, dtype="float64").ravel()
    w = np.asarray(weights, dtype="float64").ravel()
    if v.shape != w.shape:
        raise ValueError(f"shape mismatch {v.shape} vs {w.shape}")
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = w.sum()
    if total <= 0:
        return {"mean": float("nan"), "valid_frac": float("nan"), "weight": 0.0}
    ok = np.isfinite(v) & (w > 0)
    wv = w[ok].sum()
    mean = float((w[ok] * v[ok]).sum() / wv) if wv > 0 else float("nan")
    return {"mean": mean, "valid_frac": float(wv / total), "weight": float(total)}


def weighted_median(values, weights) -> float:
    """Weighted median (lower median: smallest v with cumulative weight >= half)."""
    v = np.asarray(values, dtype="float64").ravel()
    w = np.asarray(weights, dtype="float64").ravel()
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    v, w = v[ok], w[ok]
    order = np.argsort(v, kind="stable")
    cw = np.cumsum(w[order])
    i = int(np.searchsorted(cw, 0.5 * cw[-1], side="left"))
    return float(v[order][i])


def composite_dates(start: date, end: date) -> dict[str, date]:
    """mid_date (centre of the window, floored) and available_date = end + LAG_DAYS."""
    return {
        "mid_date": start + timedelta(days=(end - start).days // 2),
        "available_date": end + timedelta(days=LAG_DAYS),
    }


# --- retries / signing --------------------------------------------------------------
def _retry(fn, what: str, tries: int = _RETRIES, base: float = 5.0):
    """Call ``fn()`` with exponential backoff; on any failure clear the PC SAS-token cache
    (expired/rejected tokens surface as HTTP 403 from blob storage) so the next attempt
    re-signs the URLs."""
    import planetary_computer.sas as sas

    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if attempt == tries - 1:
                raise
            if "403" in str(e) or "Forbidden" in str(e) or "AuthenticationFailed" in str(e):
                sas.TOKEN_CACHE.clear()
            wait = base * 2**attempt
            log.warning("%s failed (%s: %s); retry %d in %.0fs", what, type(e).__name__,
                        str(e)[:200], attempt + 1, wait)
            time.sleep(wait)


def _sign_url(url: str) -> str:
    import planetary_computer as pc

    return pc.sign(url)


_rio_lock = threading.Lock()
_rio_done = False


def _configure_rio() -> None:
    global _rio_done
    with _rio_lock:
        if not _rio_done:
            import odc.stac

            odc.stac.configure_rio(
                cloud_defaults=True, GDAL_HTTP_MAX_RETRY="4", GDAL_HTTP_RETRY_DELAY="2"
            )
            _rio_done = True


# --- STAC ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _state_bbox(state: str) -> tuple[float, float, float, float]:
    from cornsat.aoi import load_states

    return tuple(float(x) for x in load_states(STATES[state]).total_bounds)


def platform_of(item) -> str:
    p = item.properties.get("platform")
    if p in PLATFORMS.values():
        return p
    return PLATFORMS[item.id.split(".")[0]]


def search_items(state: str, year: int) -> list:
    """Unsigned STAC items (Terra + Aqua) intersecting the state whose 16-day window
    overlaps SEASON_START..SEASON_END of ``year``. Duplicate (platform, start, tile)
    entries keep the most recently produced one. URLs are signed at read time."""
    import pystac_client

    s = date(year, *SEASON_START).isoformat()
    e = date(year, *SEASON_END).isoformat()

    def run():
        cat = pystac_client.Client.open(STAC_URL)
        return list(
            cat.search(collections=[COLLECTION], bbox=_state_bbox(state),
                       datetime=f"{s}/{e}", limit=500).items()
        )

    items = _retry(run, f"STAC search {state} {year}")
    best: dict[tuple, object] = {}
    for it in items:
        key = (platform_of(it), it.properties["start_datetime"],
               it.properties.get("modis:tile-id", it.id.split(".")[2]))
        if key not in best or it.id > best[key].id:
            best[key] = it
    return sorted(best.values(), key=lambda i: (i.properties["start_datetime"], i.id))


def group_composites(items) -> list[tuple[str, date, date, list]]:
    """Group items into composites: (platform, start_date, end_date, items)."""
    groups: dict[tuple[str, str, str], list] = {}
    for it in items:
        key = (platform_of(it), it.properties["start_datetime"], it.properties["end_datetime"])
        groups.setdefault(key, []).append(it)
    out = []
    for (plat, s, e), its in groups.items():
        out.append((plat, date.fromisoformat(s[:10]), date.fromisoformat(e[:10]), its))
    return sorted(out, key=lambda g: (g[1], g[0]))


# --- loading ------------------------------------------------------------------------
def load_composite(items_for_one_date, state: str, chunk: int = 1024) -> xr.Dataset:
    """Lazy (dask) Dataset with ``ndvi``, ``evi``, ``reliability`` and ``doy`` (raw values,
    raw dtypes) on ``modis_geobox(state)``, tiles mosaicked into one 2-D slice."""
    import odc.stac

    _configure_rio()
    items = list(items_for_one_date)
    starts = {i.properties["start_datetime"] for i in items}
    plats = {platform_of(i) for i in items}
    if len(starts) != 1 or len(plats) != 1:
        raise ValueError(f"items span several composites: {starts} {plats}")
    cfg = {COLLECTION: {"assets": {a: {"nodata": nd, "data_type": dt}
                                   for a, (_, nd, dt) in BANDS.items()}}}
    ds = odc.stac.load(
        items,
        bands=list(BANDS),
        geobox=modis_geobox(state),
        chunks={"x": chunk, "y": chunk},
        groupby="solar_day",
        stac_cfg=cfg,
        patch_url=_sign_url,
        fail_on_error=True,
    )
    if ds.sizes.get("time", 1) != 1:
        raise RuntimeError(f"expected one time slice, got {ds.sizes['time']}")
    return ds.rename({a: short for a, (short, _, _) in BANDS.items()}).isel(time=0, drop=True)


def _composite_row(state, year, plat, start, end, items, weight, cdl_years, workers) -> dict:
    def fetch():
        ds = load_composite(items, state)
        return ds.compute(scheduler="threads", num_workers=workers)

    ds = _retry(fetch, f"load {state} {plat} {start}")
    ndvi, evi = apply_qa(ds["ndvi"].values, ds["evi"].values, ds["reliability"].values)
    ndvi_g, evi_g = apply_qa(ds["ndvi"].values, ds["evi"].values, ds["reliability"].values,
                             keep=(0,))
    s_ndvi = weighted_stats(ndvi, weight)
    s_good = weighted_stats(ndvi_g, weight)
    doy = ds["doy"].values.astype("float64")
    doy[(doy < 1) | (doy > 366) | ~np.isfinite(ndvi)] = np.nan
    doy_mean = weighted_stats(doy, weight)["mean"]
    obs = (date(start.year, 1, 1) + timedelta(days=round(doy_mean) - 1)
           if np.isfinite(doy_mean) else None)
    if obs is not None and obs < start:  # composite crossing new year (not in season)
        obs = date(end.year, 1, 1) + timedelta(days=round(doy_mean) - 1)
    return {
        "state": state, "year": year, "platform": plat,
        "start_date": start, "end_date": end, **composite_dates(start, end),
        "obs_date": obs,
        "ndvi_mean": s_ndvi["mean"],
        "evi_mean": weighted_stats(evi, weight)["mean"],
        "ndvi_mean_good": s_good["mean"],
        "evi_mean_good": weighted_stats(evi_g, weight)["mean"],
        "ndvi_median": weighted_median(ndvi, weight),
        "valid_frac": s_ndvi["valid_frac"],
        "valid_frac_good": s_good["valid_frac"],
        "n_tiles": len(items),
        "cdl_years": ",".join(str(y) for y in cdl_years),
    }  # fmt: skip


def _to_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ["start_date", "end_date", "mid_date", "available_date", "obs_date"]:
        df[c] = pd.to_datetime(df[c])
    df["year"] = df["year"].astype("int64")
    df["n_tiles"] = df["n_tiles"].astype("int64")
    return df.sort_values(["state", "start_date", "platform"]).reset_index(drop=True)


def state_composite_stats(state: str, year: int, workers: int = 8, parallel: int = 3,
                          progress: bool = True) -> pd.DataFrame:
    """One row per (state, platform, composite) overlapping the season of ``year``.

    Columns: state, year, platform, start_date, end_date (16-day window, inclusive),
    mid_date, available_date (= end_date + LAG_DAYS; see LAG_DAYS: a conservative
    production/publication lag, the first date the backtest may use the row), obs_date
    (corn-weighted mean actual observation day from composite_day_of_the_year),
    ndvi_mean / evi_mean (corn-weighted, reliability 0 or 1), ndvi_mean_good /
    evi_mean_good (reliability 0 only), ndvi_median (weighted), valid_frac /
    valid_frac_good (share of corn weight with valid NDVI), n_tiles, cdl_years.

    ``workers`` = dask threads per composite, ``parallel`` = composites in flight.
    """
    w = season_weight(state, year)
    weight = w.values.astype("float32")
    cdl_years = list(w.attrs["cdl_years"])
    comps = group_composites(search_items(state, year))
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = [ex.submit(_composite_row, state, year, p, s, e, its, weight, cdl_years, workers)
                for p, s, e, its in comps]
        for f in tqdm(futs, desc=f"{state} {year}", disable=not progress, leave=False):
            rows.append(f.result())
    return _to_frame(rows)


def _year_path(year: int) -> Path:
    return MODIS_DIR / f"{year}.parquet"


def build_year(year: int, states=STATES, workers: int = 8, parallel: int = 3,
               refresh: bool = False) -> pd.DataFrame:
    """All states for one season, cached to data/interim/modis/{year}.parquet.

    Resumable: a cached year is returned as-is unless ``refresh``. If the cache lacks
    some requested states, only those are built and appended.
    """
    path = _year_path(year)
    cached = pd.read_parquet(path) if path.exists() else None
    todo = list(states)
    if cached is not None and not refresh:
        todo = [s for s in todo if s not in set(cached["state"])]
        if not todo:
            return cached
    frames = [] if cached is None or refresh else [cached]
    for st in todo:
        t0 = time.time()
        df = state_composite_stats(st, year, workers=workers, parallel=parallel)
        log.info("%s %d: %d composites in %.0fs", st, year, len(df), time.time() - t0)
        tqdm.write(f"{st} {year}: {len(df)} composites, {time.time() - t0:.0f}s, "
                   f"mean valid_frac {df['valid_frac'].mean():.2f}")
        frames.append(df)
    out = _to_frame(pd.concat(frames, ignore_index=True).to_dict("records"))
    MODIS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.parquet")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return out


def build_all(years=None, states=STATES, workers: int = 8, parallel: int = 3) -> pd.DataFrame:
    """Build (or read cached) years and write data/processed/modis_state.parquet."""
    from cornsat.config import YEARS

    years = list(YEARS if years is None else years)
    frames = [build_year(y, states, workers=workers, parallel=parallel)
              for y in tqdm(years, desc="years")]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["state"].isin(list(states))]
    df = df.sort_values(["state", "start_date", "platform"]).reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    return df


def load_modis_state() -> pd.DataFrame:
    """Read data/processed/modis_state.parquet."""
    return pd.read_parquet(OUT_PATH)

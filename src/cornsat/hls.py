"""NASA HLS v2.0 (HLSL30 + HLSS30): token-safe COG access (8a) and season ingestion (8c).

Access (8a): search goes through CMR (STAC for bboxes, the granule API for MGRS tiles);
reads go through GDAL /vsicurl on a pre-signed URL obtained with the bearer token. The
token, request headers and pre-signed URLs are never logged or put in error messages.

Ingestion (8c): for every season (May 1 - Sep 30, 2013-2026; 2013-2015 = L30-only
pre-sample) and every MGRS tile of the fixed block sample (``cornsat.hlssample``), each
granule with scene cloud cover <= 90% is read block-wise: Fmask first (adjacent blocks
merged into one window read), reflectance (Blue, Red, NIR) only when >= 5% of the
tile's season-mask pixels are clear. One row per (granule, block) is cached, then
``composite`` builds causal 16-day state composites every 4 days for
``features.load_hls``. See ``FMASK_POLICY`` for the pixel rules.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import rasterio
import requests
from dotenv import load_dotenv
from rasterio.windows import Window

from cornsat.config import DATA_INTERIM, DATA_PROCESSED, ROOT

log = logging.getLogger(__name__)

STAC_URL = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD"
CMR_GRANULES_URL = "https://cmr.earthdata.nasa.gov/search/granules.umm_json"
COLLECTIONS = ("HLSL30_2.0", "HLSS30_2.0")

BANDS = {
    "L30": {"blue": "B02", "red": "B04", "nir": "B05"},
    "S30": {"blue": "B02", "red": "B04", "nir": "B8A"},
}
FMASK = "Fmask"
NODATA = -9999
SCALE = 1e-4

_ID_RE = re.compile(
    r"^HLS\.(?P<sensor>[LS]30)\.T(?P<tile>\d{2}[A-Z]{3})\.(?P<date>\d{7})T(?P<time>\d{6})\.v2\.0$"
)


class EarthdataTokenError(RuntimeError):
    """Raised when EARTHDATA_TOKEN is missing or expired (never contains the value)."""


def _decode_exp(tok: str) -> float | None:
    """Best-effort JWT exp claim; None if not decodable."""
    try:
        payload = tok.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return float(json.loads(base64.urlsafe_b64decode(payload))["exp"])
    except Exception:
        return None


def token() -> str:
    load_dotenv(ROOT / ".env")
    tok = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if not tok:
        raise EarthdataTokenError(
            "EARTHDATA_TOKEN is not set. Generate one at https://urs.earthdata.nasa.gov "
            "(Generate Token) and put it in .env as EARTHDATA_TOKEN=..."
        )
    exp = _decode_exp(tok)
    if exp is not None and exp < time.time():
        raise EarthdataTokenError(
            "EARTHDATA_TOKEN has expired; generate a new one at https://urs.earthdata.nasa.gov."
        )
    return tok


def gdal_env(**extra) -> rasterio.Env:
    """rasterio.Env tuned for HTTPS COG reads. Holds no credentials.

    Auth: data.lpdaac.earthdatacloud.nasa.gov answers an Authorization: Bearer request with a
    303 to a pre-signed CloudFront URL. GDAL's GDAL_HTTP_HEADERS bearer approach did NOT open the
    COGs (verified), so `resolve_href` performs the authenticated hop with `requests` and GDAL
    reads the pre-signed URL (no secret in the URL beyond a short-lived signature).
    """
    opts = dict(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
        GDAL_HTTP_MAX_RETRY="4",
        GDAL_HTTP_RETRY_DELAY="2",
        GDAL_HTTP_TIMEOUT="60",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE=str(50_000_000),
        CPL_VSIL_CURL_CACHE_SIZE=str(200_000_000),
    )
    opts.update(extra)
    return rasterio.Env(**opts)


_RESOLVED: dict[str, tuple[str, float]] = {}
_RESOLVE_LOCK = threading.Lock()
_STATS = {"resolve_calls": 0, "http_429": 0}


def resolve_href(href: str, force: bool = False) -> str:
    """Authenticated hop: return the short-lived pre-signed URL for a data.lpdaac href (cached)."""
    now = time.time()
    with _RESOLVE_LOCK:
        hit = _RESOLVED.get(href)
    if hit and not force and hit[1] - now > 120:
        return hit[0]
    tok = token()

    def go():
        _STATS["resolve_calls"] += 1
        r = requests.get(
            href,
            headers={"Authorization": f"Bearer {tok}", "Range": "bytes=0-0"},
            allow_redirects=False,
            timeout=60,
        )
        if r.status_code == 429:
            _STATS["http_429"] += 1
        if r.status_code in (301, 302, 303, 307, 308):
            return r
        if r.status_code in (401, 403):
            raise EarthdataTokenError(
                f"Earthdata rejected the token (HTTP {r.status_code}); it may be expired or lack "
                "LP DAAC access/EULA approval."
            )
        r.raise_for_status()
        return r

    r = _retry(go)
    loc = r.headers.get("Location")
    if not loc:  # served directly, no redirect
        loc, exp = href, now + 3600
    else:
        q = parse_qs(urlparse(loc).query)
        exp = float(q.get("Expires", q.get("X-Amz-Expires", [now + 3600]))[0])
        if exp < 1e9:  # relative seconds
            exp = now + exp
    with _RESOLVE_LOCK:
        _RESOLVED[href] = (loc, exp)
    return loc


def open_cog(href: str):
    """Open a remote HLS COG (call inside `with gdal_env():`); retries once on an expired URL."""
    try:
        return rasterio.open(resolve_href(href))
    except rasterio.errors.RasterioIOError:
        return rasterio.open(resolve_href(href, force=True))


def read_window(href: str, window: Window, band: int = 1):
    """Read one window from a remote COG; returns a 2-D array (raw values)."""
    with gdal_env():
        with open_cog(href) as src:
            return src.read(band, window=window)


@dataclass(frozen=True)
class HLSItem:
    id: str
    tile: str
    sensor: str  # L30 / S30
    platform: str
    datetime: datetime
    cloud_cover: float | None
    assets: dict = field(default_factory=dict)  # blue/red/nir/fmask -> href


def parse_item(item) -> HLSItem:
    """Parse a pystac Item (or anything with .id/.properties/.assets)."""
    m = _ID_RE.match(item.id)
    if not m:
        raise ValueError(f"unrecognised HLS id: {item.id}")
    sensor, tile = m["sensor"], m["tile"]
    dt = datetime.strptime(m["date"] + m["time"], "%Y%j%H%M%S").replace(tzinfo=timezone.utc)
    props = item.properties or {}
    if props.get("datetime"):
        try:
            dt = datetime.fromisoformat(str(props["datetime"]).replace("Z", "+00:00"))
        except ValueError:
            pass
    hrefs = {}
    for name, band in BANDS[sensor].items():
        a = item.assets.get(band)
        if a is not None:
            hrefs[name] = a.href
    a = item.assets.get(FMASK)
    if a is not None:
        hrefs["fmask"] = a.href
    cc = props.get("eo:cloud_cover")
    return HLSItem(
        id=item.id,
        tile=tile,
        sensor=sensor,
        platform=props.get("platform", ""),
        datetime=dt,
        cloud_cover=None if cc is None else float(cc),
        assets=hrefs,
    )


def _retry(fn, tries: int = 5, base: float = 2.0):
    for i in range(tries):
        try:
            return fn()
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if code not in (429, 500, 502, 503, 504) or i == tries - 1:
                raise
        except (requests.ConnectionError, requests.Timeout):
            if i == tries - 1:
                raise
        time.sleep(base * 2**i)


def _stac_search(**kw):
    """pystac-client search with urllib3 retries on 429/5xx."""
    from pystac_client import Client
    from pystac_client.stac_api_io import StacApiIO
    from urllib3 import Retry

    io = StacApiIO(
        timeout=60,
        max_retries=Retry(
            total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=None
        ),
    )
    client = Client.open(STAC_URL, stac_io=io)
    return client.search(**kw)


def search(tile_or_bbox, start, end, collections=COLLECTIONS, max_items: int = 5000) -> list[HLSItem]:
    """Search HLS granules by bbox [w,s,e,n] (CMR STAC) or MGRS tile 'T15TVG' (CMR granules).

    The STAC endpoint does not support CQL2 id filters, so tile queries go through the
    CMR granule API (readable_granule_name pattern) and are parsed into the same HLSItem.
    """
    if isinstance(tile_or_bbox, str):
        tile = tile_or_bbox.lstrip("T")
        items = []
        for coll in collections:
            for g in cmr_granules(coll.split("_")[0], start, end, tile=tile):
                items.append(_item_from_umm(g))
        return sorted(items, key=lambda i: i.datetime)
    kw = dict(
        collections=list(collections),
        datetime=f"{start}/{end}",
        max_items=max_items,
        limit=200,
        bbox=list(tile_or_bbox),
    )
    items = []
    for it in _stac_search(**kw).items():
        try:
            items.append(parse_item(it))
        except ValueError:
            continue
    return sorted(items, key=lambda i: i.datetime)


class _A:
    def __init__(self, href):
        self.href = href


class _Fake:
    def __init__(self, id, properties, assets):
        self.id, self.properties, self.assets = id, properties, assets


def _item_from_umm(g: dict) -> HLSItem:
    assets = {}
    for u in g["urls"]:
        m = re.search(r"\.(B\d[\dA]|Fmask)\.tif$", u)
        if m:
            assets[m.group(1)] = _A(u)
    props = {"datetime": g["acquisition"], "eo:cloud_cover": g.get("cloud_cover"),
             "platform": g.get("platform", "")}
    return parse_item(_Fake(g["id"], props, assets))


def _first_float(vals) -> float | None:
    try:
        return float(vals[0]) if vals else None
    except (TypeError, ValueError):
        return None


def cmr_granules(short_name: str, start, end, bbox=None, tile: str | None = None, page_size: int = 2000):
    """CMR granule search (UMM-JSON) -> dicts: id, acquisition, production, revision, cloud_cover,
    spatial_coverage, platform, urls."""
    params = {
        "short_name": short_name,
        "version": "2.0",
        "temporal": f"{start}T00:00:00Z,{end}T23:59:59Z",
        "page_size": page_size,
    }
    if bbox is not None:
        params["bounding_box"] = ",".join(str(x) for x in bbox)
    if tile:
        params["readable_granule_name"] = f"HLS.*.T{tile.lstrip('T')}.*"
        params["options[readable_granule_name][pattern]"] = "true"
    out, headers = [], {}
    while True:

        def go():
            r = requests.get(CMR_GRANULES_URL, params=params, headers=headers, timeout=60)
            r.raise_for_status()
            return r

        r = _retry(go)
        items = r.json().get("items", [])
        for it in items:
            umm, meta = it["umm"], it["meta"]
            prod = (umm.get("DataGranule") or {}).get("ProductionDateTime")
            rng = (umm.get("TemporalExtent") or {}).get("RangeDateTime") or {}
            attrs = {a["Name"]: a["Values"] for a in umm.get("AdditionalAttributes", [])}
            cc = attrs.get("CLOUD_COVERAGE")
            out.append(
                dict(
                    id=umm["GranuleUR"],
                    acquisition=rng.get("BeginningDateTime"),
                    production=prod,
                    revision=meta.get("revision-date"),
                    revision_id=meta.get("revision-id"),
                    cloud_cover=float(cc[0]) if cc else None,
                    spatial_coverage=_first_float(attrs.get("SPATIAL_COVERAGE")),
                    platform=((umm.get("Platforms") or [{}])[0]).get("ShortName", ""),
                    urls=[
                        x["URL"]
                        for x in umm.get("RelatedUrls", [])
                        if x["URL"].startswith("https://data.lpdaac")
                    ],
                )
            )
        sa = r.headers.get("CMR-Search-After")
        if not sa or not items:
            break
        headers["CMR-Search-After"] = sa
    return out


# =====================================================================================
# Phase 8c: season ingestion (block x granule statistics) and causal state composites
# =====================================================================================
#
# Data flow
#   CMR granule list per (year, tile)       -> data/interim/hls/items/{year}_{tile}.parquet
#   per granule: Fmask, then Blue/Red/NIR   -> data/interim/hls/scenes/{year}/{tile}.parquet
#     (one row per (granule, block); only the sampled 256 x 256 blocks of hlssample are read)
#   composite(): causal 16-day composites    -> data/processed/hls_state.parquet (2016-2026)
#     ending every 4 days (fixed DOY grid)      data/processed/hls_state_presample.parquet
#                                               (2013-2015, L30 only, climatology only)


HLS_DIR = DATA_INTERIM / "hls"
ITEMS_DIR = HLS_DIR / "items"
SCENES_DIR = HLS_DIR / "scenes"
HLS_STATE_PATH = DATA_PROCESSED / "hls_state.parquet"
HLS_PRESAMPLE_PATH = DATA_PROCESSED / "hls_state_presample.parquet"

SEASON_WINDOW = ((5, 1), (9, 30))  # acquisitions read each season (May 1 - Sep 30)
FIRST_YEAR = 2013  # Landsat-8 (L30) from 2013-04
PRESAMPLE_YEARS = (2013, 2014, 2015)  # climatology only; L30 only (S30 starts 2015-12)
SAMPLE_FIRST_YEAR = 2016
MAX_SCENE_CLOUD = 90.0  # scene-level prefilter (%): sampled blocks can be clear in cloudy scenes
MIN_TILE_CLEAR = 0.05  # skip reflectance if < 5% of the tile's season-mask pixels are clear
MIN_BLOCK_CLEAR = 0.5  # block-scene enters a composite only if n_clear / n_mask >= 0.5
COMPOSITE_WINDOW_DAYS = 16  # [E - 15, E]
COMPOSITE_STEP_DAYS = 4
COMPOSITE_FIRST = (5, 16)  # first composite end date (calendar DOY 136)
COMPOSITE_LAST = (9, 30)  # no composite ends after Sep 30
DEFAULT_WORKERS = 32
FLUSH_SECONDS = 60.0

# --- Fmask policy (HLS v2.0 Fmask, uint8) -------------------------------------------
# bit 0 cirrus | 1 cloud | 2 adjacent to cloud/shadow | 3 cloud shadow | 4 snow/ice |
# 5 water | 6-7 aerosol level (0 climatology, 1 low, 2 moderate, 3 high); 255 = nodata.
FMASK_NODATA = 255
FMASK_REJECT_BITS = 0b0011_1111  # any of bits 0-5 set -> reject
AEROSOL_SHIFT = 6
AEROSOL_HIGH = 3  # aerosol level 3 (high) -> reject; levels 0-2 are kept
FMASK_POLICY = (
    "reject Fmask==255 (nodata), any of bits 0-5 (cirrus, cloud, adjacent cloud/shadow, "
    "cloud shadow, snow/ice, water), aerosol bits 6-7 == 3 (high); reject reflectance "
    "-9999 or any band <= 0 or > 1 after x1e-4; reject |EVI| > 1 or NDVI outside [-1, 1]"
)

SCENE_COLS = [
    "granule_id", "tile", "sensor", "platform", "acq_dt", "cloud_cover", "block_id",
    "state", "season", "n_mask", "n_footprint", "n_clear", "ndvi_mean", "evi_mean",
    "ndvi_median", "refl_read",
]  # fmt: skip
COMPOSITE_COLS = [
    "state", "year", "platform", "start_date", "end_date", "mid_date", "obs_date",
    "available_date", "ndvi_mean", "evi_mean", "ndvi_median", "valid_frac", "n_tiles",
    "n_blocks", "n_pixels", "n_l30", "n_s30", "cdl_years",
]  # fmt: skip


# --- pixel rules ----------------------------------------------------------------------
def fmask_clear(fm: np.ndarray) -> np.ndarray:
    """True where an Fmask pixel is usable under ``FMASK_POLICY``."""
    fm = np.asarray(fm).astype(np.uint8, copy=False)
    aerosol = (fm >> AEROSOL_SHIFT) & 0b11
    return (fm != FMASK_NODATA) & ((fm & FMASK_REJECT_BITS) == 0) & (aerosol != AEROSOL_HIGH)


def band_names(sensor: str) -> dict[str, str]:
    """{'blue','red','nir'} -> HLS asset band name for a sensor (L30 NIR = B05, S30 = B8A)."""
    return dict(BANDS[sensor])


def scale_reflectance(raw: np.ndarray) -> np.ndarray:
    """int16 surface reflectance -> float (x SCALE); NODATA (-9999) -> NaN."""
    raw = np.asarray(raw)
    out = raw.astype("float32") * np.float32(SCALE)
    out[raw == NODATA] = np.nan
    return out


def vegetation_indices(blue, red, nir):
    """NDVI, EVI and a validity mask from scaled reflectance (NaN = nodata).

    Valid: every band finite and in (0, 1], NDVI in [-1, 1], |EVI| <= 1 (and finite).
    NDVI = (N-R)/(N+R); EVI = 2.5 (N-R) / (N + 6R - 7.5B + 1).
    """
    b, r, n = (np.asarray(x, dtype="float64") for x in (blue, red, nir))
    with np.errstate(invalid="ignore", divide="ignore"):
        ok = np.isfinite(b) & np.isfinite(r) & np.isfinite(n)
        ok &= (b > 0) & (b <= 1) & (r > 0) & (r <= 1) & (n > 0) & (n <= 1)
        ndvi = (n - r) / (n + r)
        evi = 2.5 * (n - r) / (n + 6.0 * r - 7.5 * b + 1.0)
        ok &= np.isfinite(ndvi) & np.isfinite(evi) & (ndvi >= -1) & (ndvi <= 1) & (np.abs(evi) <= 1)
    return ndvi, evi, ok


def block_stats(fmask, mask, blue_raw=None, red_raw=None, nir_raw=None) -> dict:
    """Statistics of one block in one granule.

    n_mask = season-mask pixels; n_footprint = mask pixels inside the granule footprint
    (Fmask != 255); n_clear = mask & Fmask-clear (& valid reflectance and indices when the
    bands are given). Without bands the values are NaN (cloudy-skipped granule).
    """
    mask = np.asarray(mask, bool)
    fm = np.asarray(fmask)
    sel = mask & fmask_clear(fm)
    out = {
        "n_mask": int(mask.sum()),
        "n_footprint": int((mask & (fm != FMASK_NODATA)).sum()),
        "n_clear": int(sel.sum()),
        "ndvi_mean": np.nan,
        "evi_mean": np.nan,
        "ndvi_median": np.nan,
    }
    if blue_raw is None:
        return out
    if not sel.any():
        out["n_clear"] = 0
        return out
    ndvi, evi, ok = vegetation_indices(
        scale_reflectance(np.asarray(blue_raw)[sel]),
        scale_reflectance(np.asarray(red_raw)[sel]),
        scale_reflectance(np.asarray(nir_raw)[sel]),
    )
    out["n_clear"] = int(ok.sum())
    if ok.any():
        out["ndvi_mean"] = float(ndvi[ok].mean())
        out["evi_mean"] = float(evi[ok].mean())
        out["ndvi_median"] = float(np.median(ndvi[ok]))
    return out


# --- window merging ---------------------------------------------------------------------
def merge_windows(offsets, block: int = 256) -> list[tuple[Window, list[int]]]:
    """Group block-aligned (row_off, col_off) blocks into 4-connected components and return
    one bounding Window per component with the member indices (fewer range requests: GDAL
    fetches the internal tiles of one window read in a single multi-range round)."""
    cells: dict[tuple[int, int], int] = {}
    for i, (r, c) in enumerate(offsets):
        if r % block or c % block:
            raise ValueError("block offsets must be multiples of the block size")
        cells[(r // block, c // block)] = i
    seen: set = set()
    out = []
    for cell in sorted(cells):
        if cell in seen:
            continue
        seen.add(cell)
        stack, comp = [cell], []
        while stack:
            a = stack.pop()
            comp.append(a)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                b = (a[0] + dr, a[1] + dc)
                if b in cells and b not in seen:
                    seen.add(b)
                    stack.append(b)
        r0 = min(x[0] for x in comp)
        r1 = max(x[0] for x in comp)
        c0 = min(x[1] for x in comp)
        c1 = max(x[1] for x in comp)
        win = Window(c0 * block, r0 * block, (c1 - c0 + 1) * block, (r1 - r0 + 1) * block)
        out.append((win, sorted(cells[x] for x in comp)))
    return out


# --- remote reads with counters ---------------------------------------------------------
class IngestStats:
    """Thread-safe counters (no URLs are ever stored)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.c: dict[str, int] = {}

    def add(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.c[key] = self.c.get(key, 0) + n

    def get(self, key: str) -> int:
        return self.c.get(key, 0)


STATS = IngestStats()


MAX_INFLIGHT = 128  # global cap on concurrent network operations (resolve / open / window)
_INFLIGHT = threading.BoundedSemaphore(MAX_INFLIGHT)


def _net(fn, *a, **k):
    with _INFLIGHT:
        return fn(*a, **k)


def _read_one(url: str, window: Window) -> np.ndarray:
    # re-opening a URL already opened in this process is served from GDAL's /vsicurl cache
    with gdal_env(), rasterio.open(url) as src:
        return src.read(1, window=window)


def _read_one_net(url: str, window: Window) -> np.ndarray:
    return _net(_read_one, url, window)


def read_windows(href: str, windows, tries: int = 5, base: float = 2.0,
                 parallel: bool = True) -> list[np.ndarray]:
    """Read several windows (band 1) of one remote COG.

    One resolve + one open (which caches the COG header), then the windows are read
    concurrently (each through its own handle on the cached header) when ``parallel``.
    Exponential backoff with jitter on any I/O failure (GDAL itself already retries
    429/5xx via GDAL_HTTP_MAX_RETRY); every retry re-resolves the pre-signed URL, which
    handles expiry. Errors are re-raised without the URL in the message.
    """
    windows = list(windows)
    for i in range(tries):
        try:
            url = _net(resolve_href, href, force=i > 0)
            with gdal_env(), _INFLIGHT, rasterio.open(url) as src:
                STATS.add("opens")
                if not parallel or len(windows) <= 1:
                    out = [src.read(1, window=w) for w in windows]
                else:
                    out = None
            if out is None:
                with ThreadPoolExecutor(max_workers=len(windows)) as ex:
                    out = list(ex.map(_read_one_net, [url] * len(windows), windows))
            STATS.add("window_reads", len(windows))
            return out
        except EarthdataTokenError:
            raise
        except Exception as e:  # noqa: BLE001
            STATS.add("retries")
            if i == tries - 1:
                raise RuntimeError(f"COG read failed after {tries} tries ({type(e).__name__})") from None
            time.sleep(base * 2**i * (0.75 + 0.5 * random.random()))
    raise AssertionError("unreachable")


# --- granule list -------------------------------------------------------------------------
def _items_path(year: int, tile: str) -> Path:
    return ITEMS_DIR / f"{year}_{tile.lstrip('T')}.parquet"


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"refusing to write through a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def season_dates(year: int) -> tuple[date, date]:
    (m0, d0), (m1, d1) = SEASON_WINDOW
    return date(year, m0, d0), date(year, m1, d1)


def sensors_for(year: int) -> tuple[str, ...]:
    """L30 only for the 2013-2015 pre-sample, L30 + S30 from 2016."""
    return ("L30",) if year < SAMPLE_FIRST_YEAR else ("L30", "S30")


def granule_items(year: int, tile: str, refresh: bool = False) -> pd.DataFrame:
    """All HLS granules of an MGRS tile in the season window (CMR), cached per (year, tile).

    Columns: id, tile, sensor, platform, acq_dt (UTC), cloud_cover, spatial_coverage,
    production, href_blue/red/nir/fmask (data.lpdaac paths, not pre-signed). The cache is
    refreshed if it was written before Oct 31 of ``year`` (late-processed granules).
    """
    tile = tile.lstrip("T")
    path = _items_path(year, tile)
    stale_before = pd.Timestamp(year, 10, 31).timestamp()
    if path.exists() and not refresh and path.stat().st_mtime >= stale_before:
        return pd.read_parquet(path)
    start, end = season_dates(year)
    rows = []
    for sensor in sensors_for(year):
        for g in cmr_granules(f"HLS{sensor}", start.isoformat(), end.isoformat(), tile=tile):
            m = _ID_RE.match(g["id"])
            if not m or m["tile"] != tile:
                continue
            hrefs = {}
            for u in g["urls"]:
                mm = re.search(r"\.(B\d[\dA]|Fmask)\.tif$", u)
                if mm:
                    hrefs[mm.group(1)] = u
            names = {**{k: v for k, v in BANDS[sensor].items()}, "fmask": FMASK}
            if not all(v in hrefs for v in names.values()):
                continue
            acq = g.get("acquisition")
            dt = (pd.Timestamp(acq) if acq else
                  pd.Timestamp(datetime.strptime(m["date"] + m["time"], "%Y%j%H%M%S"), tz="UTC"))
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            rows.append({
                "id": g["id"], "tile": "T" + tile, "sensor": sensor,
                "platform": g.get("platform", ""), "acq_dt": dt.tz_convert("UTC"),
                "cloud_cover": g.get("cloud_cover"), "spatial_coverage": g.get("spatial_coverage"),
                "production": g.get("production"),
                **{f"href_{k}": hrefs[v] for k, v in names.items()},
            })  # fmt: skip
    cols = ["id", "tile", "sensor", "platform", "acq_dt", "cloud_cover", "spatial_coverage",
            "production", "href_blue", "href_red", "href_nir", "href_fmask"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.drop_duplicates("id").sort_values("acq_dt").reset_index(drop=True)
    df["acq_dt"] = pd.to_datetime(df["acq_dt"], utc=True)
    df["cloud_cover"] = df["cloud_cover"].astype("float64")
    df["spatial_coverage"] = df["spatial_coverage"].astype("float64")
    _write_atomic(df, path)
    return df


def prefilter_items(items: pd.DataFrame, year: int, max_cloud: float = MAX_SCENE_CLOUD) -> pd.DataFrame:
    """Scene-level prefilter: allowed sensors for the year, cloud_cover <= max_cloud (unknown
    cloud cover is kept), acquisition inside the season window."""
    start, end = season_dates(year)
    d = pd.to_datetime(items["acq_dt"], utc=True).dt.tz_convert(None).dt.normalize()
    ok = items["sensor"].isin(sensors_for(year))
    ok &= items["cloud_cover"].isna() | (items["cloud_cover"] <= max_cloud)
    ok &= (d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end))
    return items[ok].reset_index(drop=True)


# --- per-granule processing ---------------------------------------------------------------
def _slice(arr, win: Window, b: dict) -> np.ndarray:
    r = int(b["row_off"]) - int(win.row_off)
    c = int(b["col_off"]) - int(win.col_off)
    return arr[r : r + int(b["height"]), c : c + int(b["width"])]


def process_granule(item, blocks: list[dict], reader=None, min_tile_clear: float = MIN_TILE_CLEAR):
    """Rows (one per block) for one granule; Fmask first, reflectance only if worth it.

    ``item``: mapping with id, tile, sensor, platform, acq_dt, cloud_cover, href_*.
    ``blocks``: dicts with block_id, state, season, row_off, col_off, height, width, mask.
    ``reader(href, windows) -> list[array]`` (default ``read_windows``).
    Reflectance is skipped when < ``min_tile_clear`` of the union of the blocks' mask
    pixels is clear; otherwise Blue/Red/NIR are read only for windows holding a block with
    at least one clear mask pixel (others cannot produce a value). Returns (rows, refl_read).
    """
    reader = reader or read_windows
    groups = merge_windows([(int(b["row_off"]), int(b["col_off"])) for b in blocks])
    wins = [g[0] for g in groups]
    fm_arrs = reader(item["href_fmask"], wins)
    fm: list = [None] * len(blocks)
    for (win, idx), arr in zip(groups, fm_arrs):
        for i in idx:
            fm[i] = _slice(arr, win, blocks[i])
    fstats = [block_stats(fm[i], b["mask"]) for i, b in enumerate(blocks)]
    tot = sum(s["n_mask"] for s in fstats)
    clear = sum(s["n_clear"] for s in fstats)
    refl_read = tot > 0 and clear / tot >= min_tile_clear
    stats = list(fstats)
    read_blocks: set[int] = set()
    if refl_read:
        need = [g for g, (_, idx) in enumerate(groups) if any(fstats[i]["n_clear"] > 0 for i in idx)]
        names = ("blue", "red", "nir")
        with ThreadPoolExecutor(max_workers=len(names)) as ex:  # the three bands concurrently
            got = list(ex.map(lambda n: reader(item[f"href_{n}"], [wins[g] for g in need]), names))
        bands = {}
        for name, arrs in zip(names, got):
            per = {}
            for g, arr in zip(need, arrs):
                win, idx = groups[g]
                for i in idx:
                    per[i] = _slice(arr, win, blocks[i])
            bands[name] = per
        for g in need:
            for i in groups[g][1]:
                b = blocks[i]
                stats[i] = block_stats(fm[i], b["mask"], bands["blue"][i], bands["red"][i],
                                       bands["nir"][i])
                read_blocks.add(i)
    acq = pd.Timestamp(item["acq_dt"])
    acq = acq.tz_localize("UTC") if acq.tzinfo is None else acq.tz_convert("UTC")
    rows = []
    for i, b in enumerate(blocks):
        rows.append({
            "granule_id": item["id"], "tile": item["tile"], "sensor": item["sensor"],
            "platform": item.get("platform", "") or "", "acq_dt": acq,
            "cloud_cover": float(item["cloud_cover"]) if item.get("cloud_cover") is not None
            else np.nan,
            "block_id": b["block_id"], "state": b["state"], "season": int(b["season"]),
            **stats[i], "refl_read": i in read_blocks,
        })  # fmt: skip
    return rows, refl_read


# --- scene cache ----------------------------------------------------------------------------
def _scenes_path(year: int, tile: str) -> Path:
    return SCENES_DIR / str(year) / f"{tile if tile.startswith('T') else 'T' + tile}.parquet"


def _empty_scenes() -> pd.DataFrame:
    return pd.DataFrame(columns=SCENE_COLS)


def load_scenes(year: int, tile: str) -> pd.DataFrame:
    p = _scenes_path(year, tile)
    return pd.read_parquet(p) if p.exists() else _empty_scenes()


def _merge_scene_rows(old: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(new_rows, columns=SCENE_COLS)
    parts = [x for x in (old, new) if len(x)]
    if not parts:
        return _empty_scenes()
    df = pd.concat(parts, ignore_index=True)
    df["acq_dt"] = pd.to_datetime(df["acq_dt"], utc=True)
    df = df.drop_duplicates(["granule_id", "block_id"], keep="last")
    for c in ("n_mask", "n_footprint", "n_clear", "season"):
        df[c] = df[c].astype("int64")
    df["refl_read"] = df["refl_read"].astype(bool)
    return df.sort_values(["acq_dt", "granule_id", "block_id"]).reset_index(drop=True)


def load_scene_rows(years=None, states=None) -> pd.DataFrame:
    """All cached (granule, block) rows, optionally restricted to seasons / states."""
    if years is None:
        years = sorted(int(p.name) for p in SCENES_DIR.glob("*") if p.name.isdigit())
    parts = []
    for y in years:
        for p in sorted((SCENES_DIR / str(y)).glob("*.parquet")):
            parts.append(pd.read_parquet(p))
    if not parts:
        return _empty_scenes()
    df = pd.concat(parts, ignore_index=True)
    if states is not None:
        sts = [states] if isinstance(states, str) else list(states)
        df = df[df["state"].isin(sts)]
    return df.reset_index(drop=True)


# --- masks ------------------------------------------------------------------------------------
class MissingMasksError(FileNotFoundError):
    pass


def _states_list(states) -> list[str]:
    from cornsat.config import STATES

    if states is None:
        return list(STATES)
    return [states] if isinstance(states, str) else list(states)


def require_masks(year: int, states=None) -> None:
    """Raise MissingMasksError unless every state's season mask is cached (never builds)."""
    from cornsat import hlssample

    missing = [st for st in _states_list(states) if not hlssample._mask_path(st, year).exists()]
    if missing:
        raise MissingMasksError(
            f"HLS season masks missing for season {year}: {', '.join(missing)}. Build them with "
            f"`uv run python scripts/build_hls_sample.py --states {' '.join(missing)} "
            f"--seasons {year}`."
        )


def season_blocks(year: int, states=None) -> pd.DataFrame:
    """Sampled blocks of ``states`` with their season-``year`` mask and n_mask (masks must
    already be cached; see ``require_masks``)."""
    from cornsat import hlssample

    sts = _states_list(states)
    require_masks(year, sts)
    b = hlssample.load_blocks(sts).copy()
    b["mask"] = [hlssample.season_mask(bid, year) for bid in b["block_id"]]
    b["n_mask"] = [int(m.sum()) for m in b["mask"]]
    b["season"] = int(year)
    return b


def cdl_years_str(state: str, year: int) -> str:
    from cornsat import hlssample

    return ",".join(str(y) for y in sorted(hlssample.mask_meta(state, year)["cdl_years"]))


# --- builder ----------------------------------------------------------------------------------
def plan_year(year: int, blocks: pd.DataFrame, items_fn=None) -> tuple[list, dict]:
    """Work list [(tile, item dict, [block dicts still missing])] plus counts. A granule is
    skipped when all its (granule, block) rows are cached (cloudy skips included)."""
    items_fn = items_fn or granule_items
    blocks = blocks[blocks["n_mask"] > 0]
    todo, counts = [], {"granules_found": 0, "granules_prefiltered": 0, "granules_cached": 0}
    tiles = list(dict.fromkeys(blocks["tile"]))

    def fetch(tile):
        return tile, items_fn(year, tile)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(tiles)))) as ex:
        fetched = dict(ex.map(fetch, tiles))
    for tile in tiles:
        tb = blocks[blocks["tile"] == tile].to_dict("records")
        items = fetched[tile]
        counts["granules_found"] += len(items)
        items = prefilter_items(items, year)
        counts["granules_prefiltered"] += len(items)
        old = load_scenes(year, tile)
        done = set(zip(old["granule_id"], old["block_id"]))
        for it in items.to_dict("records"):
            sub = [b for b in tb if (it["id"], b["block_id"]) not in done]
            if sub:
                todo.append((tile, it, sub))
            else:
                counts["granules_cached"] += 1
    return todo, counts


def build_year(year: int, states=None, workers: int = DEFAULT_WORKERS, reader=None,
               items_fn=None, progress: bool = True) -> dict:
    """Ingest one season (May 1 - Sep 30) for ``states``: resumable, one scene file per tile.

    Requires the season masks (raises MissingMasksError). Returns a summary dict with
    granule counts, request counters and elapsed seconds (never URLs).
    """
    t0 = time.time()
    blocks = season_blocks(year, states)
    todo, counts = plan_year(year, blocks, items_fn)
    before = dict(STATS.c)
    before_resolve = _STATS["resolve_calls"]
    buffers: dict[str, list] = {}
    summary = {"year": year, "states": _states_list(states), "tiles": blocks["tile"].nunique(),
               "blocks": int((blocks["n_mask"] > 0).sum()), **counts, "granules_todo": len(todo),
               "granules_read": 0, "granules_skipped_cloudy": 0, "granules_failed": 0,
               "errors": {}}  # fmt: skip

    def flush():
        for tile, rows in list(buffers.items()):
            if rows:
                _write_atomic(_merge_scene_rows(load_scenes(year, tile), rows), _scenes_path(year, tile))
                buffers[tile] = []

    bar = None
    if progress and todo:
        from tqdm import tqdm

        bar = tqdm(total=len(todo), desc=f"HLS {year}", unit="granule", smoothing=0.05)
    last = time.time()
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_granule, it, sub, reader): (tile, it["id"])
                    for tile, it, sub in todo}
            for fut in as_completed(futs):
                tile, _gid = futs[fut]
                try:
                    rows, refl = fut.result()
                except EarthdataTokenError:
                    for f in futs:
                        f.cancel()
                    raise
                except Exception as e:  # noqa: BLE001 - never print the message (URLs)
                    summary["granules_failed"] += 1
                    k = type(e).__name__
                    summary["errors"][k] = summary["errors"].get(k, 0) + 1
                else:
                    buffers.setdefault(tile, []).extend(rows)
                    summary["granules_read" if refl else "granules_skipped_cloudy"] += 1
                if bar is not None:
                    bar.update(1)
                if time.time() - last > FLUSH_SECONDS:
                    flush()
                    last = time.time()
    finally:
        flush()
        if bar is not None:
            bar.close()
    after = STATS.c
    d = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    summary["resolve_requests"] = _STATS["resolve_calls"] - before_resolve
    summary["opens"] = d.get("opens", 0)
    summary["window_reads"] = d.get("window_reads", 0)
    summary["retries"] = d.get("retries", 0)
    summary["requests"] = summary["resolve_requests"] + summary["opens"] + summary["window_reads"]
    summary["elapsed_s"] = time.time() - t0
    return summary


# --- composites --------------------------------------------------------------------------------
def _from_cal_doy(year: int, doy: int) -> date:
    """Inverse of features.cal_doy (365-day calendar; leap years skip Feb 29 after DOY 59)."""
    d = date(year, 1, 1) + timedelta(days=int(doy) - 1)
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return d + timedelta(days=1) if leap and doy >= 60 else d


def _cal_doy(d: date) -> int:
    leap = d.year % 4 == 0 and (d.year % 100 != 0 or d.year % 400 == 0)
    n = d.timetuple().tm_yday
    return n - 1 if leap and d.month > 2 else n


def composite_end_dates(year: int) -> list[date]:
    """End dates every COMPOSITE_STEP_DAYS on the 365-day DOY grid from May 16 (DOY 136) up
    to Sep 30: identical calendar dates every year (May 16, 20, ..., Sep 29)."""
    d0 = _cal_doy(date(2001, *COMPOSITE_FIRST))
    d1 = _cal_doy(date(2001, *COMPOSITE_LAST))
    return [_from_cal_doy(year, d) for d in range(d0, d1 + 1, COMPOSITE_STEP_DAYS)]


def weighted_median(values, weights) -> float:
    """Lower weighted median (smallest v with cumulative weight >= half the total)."""
    v = np.asarray(values, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    v, w = v[ok], w[ok]
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def _acq_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_convert(None).dt.normalize().astype("datetime64[ns]")


def gate_scene_rows(scene_rows: pd.DataFrame, min_clear_frac: float = MIN_BLOCK_CLEAR) -> pd.DataFrame:
    """Block-scene rows usable in a composite: n_clear / n_mask >= min_clear_frac and finite
    NDVI/EVI. Adds ``acq_date`` (UTC calendar date)."""
    sr = scene_rows
    nm = sr["n_mask"].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(nm > 0, sr["n_clear"].astype(float) / nm, 0.0)
    ok = (frac >= min_clear_frac) & np.isfinite(sr["ndvi_mean"].astype(float)) & np.isfinite(
        sr["evi_mean"].astype(float))
    out = sr[ok].copy()
    out["acq_date"] = _acq_date(out["acq_dt"])
    return out


def _block_info_from_scenes(scene_rows: pd.DataFrame) -> pd.DataFrame:
    return (scene_rows.groupby(["season", "block_id"], as_index=False)
            .agg(state=("state", "first"), tile=("tile", "first"), n_mask=("n_mask", "max")))


def composite(scene_rows: pd.DataFrame, block_info: pd.DataFrame | None = None,
              cdl_years: dict | None = None, s30_offset: dict | None = None,
              min_clear_frac: float = MIN_BLOCK_CLEAR, lag_days: int | None = None) -> pd.DataFrame:
    """Causal state composites from (granule, block) rows.

    For each end date E of ``composite_end_dates`` the window is [E-15, E] and only
    acquisitions with date <= E enter (a later scene never changes a composite ending
    earlier). Block value = median over the window's scenes of block-scene means passing the
    ``min_clear_frac`` gate; state value = n_mask-weighted mean of block values; valid_frac
    = covered n_mask / the state's total n_mask; ndvi_median = n_mask-weighted median of
    block NDVI. obs_date = n_clear-weighted mean acquisition date of the contributing
    block-scenes. available_date = E + features.HLS_LAG_DAYS. Composites without data are
    kept with NaN values and valid_frac 0 (``features.prepare_sat`` drops them).

    ``block_info`` (season, block_id, state, tile, n_mask) defaults to what the scene rows
    show; ``cdl_years`` maps (state, season) -> str; ``s30_offset`` maps season ->
    {"ndvi": d, "evi": d} subtracted from S30 block-scene values (see ``prior_s30_offsets``).
    """
    if lag_days is None:
        from cornsat.features import HLS_LAG_DAYS

        lag_days = HLS_LAG_DAYS
    bi = _block_info_from_scenes(scene_rows) if block_info is None else block_info
    g = gate_scene_rows(scene_rows, min_clear_frac)
    if s30_offset:
        g = g.copy()
        for y, off in s30_offset.items():
            m = (g["sensor"] == "S30") & (g["season"] == y)
            g.loc[m, "ndvi_mean"] -= off.get("ndvi", 0.0)
            g.loc[m, "ndvi_median"] -= off.get("ndvi", 0.0)
            g.loc[m, "evi_mean"] -= off.get("evi", 0.0)
    rows = []
    for (st, y), info in bi.groupby(["state", "season"], sort=True):
        w_all = info.set_index("block_id")["n_mask"].astype(float)
        tot = float(w_all.sum())
        gs = g[(g["state"] == st) & (g["season"] == y)]
        acq = gs["acq_date"].to_numpy("datetime64[D]")
        cy = (cdl_years or {}).get((st, int(y)), "")
        for E in composite_end_dates(int(y)):
            e = np.datetime64(E, "D")
            s = e - np.timedelta64(COMPOSITE_WINDOW_DAYS - 1, "D")
            sub = gs[(acq >= s) & (acq <= e)]
            sub = sub[sub["block_id"].isin(w_all.index)]
            rec = {"state": st, "year": int(y), "platform": "hls",
                   "start_date": pd.Timestamp(s), "end_date": pd.Timestamp(e),
                   "mid_date": pd.Timestamp(s + np.timedelta64(7, "D")),
                   "available_date": pd.Timestamp(e + np.timedelta64(lag_days, "D")),
                   "cdl_years": cy}  # fmt: skip
            if len(sub) and tot > 0:
                bv = sub.groupby("block_id").agg(ndvi=("ndvi_mean", "median"), evi=("evi_mean", "median"))
                w = w_all.reindex(bv.index)
                ws = float(w.sum())
                days = sub["acq_date"].to_numpy("datetime64[D]").astype("int64").astype(float)
                nc = sub["n_clear"].to_numpy(float)
                obs = np.datetime64(round((days * nc).sum() / nc.sum()), "D")
                rec.update(
                    obs_date=pd.Timestamp(obs),
                    ndvi_mean=float((bv["ndvi"] * w).sum() / ws),
                    evi_mean=float((bv["evi"] * w).sum() / ws),
                    ndvi_median=weighted_median(bv["ndvi"], w),
                    valid_frac=ws / tot,
                    n_tiles=int(sub["tile"].nunique()), n_blocks=len(bv),
                    n_pixels=int(nc.sum()),
                    n_l30=int(sub.loc[sub["sensor"] == "L30", "granule_id"].nunique()),
                    n_s30=int(sub.loc[sub["sensor"] == "S30", "granule_id"].nunique()),
                )  # fmt: skip
            else:
                rec.update(obs_date=pd.NaT, ndvi_mean=np.nan, evi_mean=np.nan, ndvi_median=np.nan,
                           valid_frac=0.0, n_tiles=0, n_blocks=0, n_pixels=0, n_l30=0, n_s30=0)
            rows.append(rec)
    out = pd.DataFrame(rows, columns=COMPOSITE_COLS)
    for c in ("start_date", "end_date", "mid_date", "obs_date", "available_date"):
        out[c] = pd.to_datetime(out[c]).astype("datetime64[ns]")
    for c in ("year", "n_tiles", "n_blocks", "n_pixels", "n_l30", "n_s30"):
        out[c] = out[c].astype("int64")
    out["cdl_years"] = out["cdl_years"].astype(str)
    return out


# --- cross-sensor diagnostic ---------------------------------------------------------------
def sensor_pairs(scene_rows: pd.DataFrame, max_days: int = 1,
                 min_clear_frac: float = MIN_BLOCK_CLEAR) -> pd.DataFrame:
    """L30 vs S30 observations of the same block <= ``max_days`` apart (both passing the
    block gate); each L30 block-scene is paired with its nearest S30 block-scene.
    dndvi / devi = S30 - L30."""
    g = gate_scene_rows(scene_rows, min_clear_frac)
    cols = ["block_id", "season", "granule_id", "acq_date", "ndvi_mean", "evi_mean"]
    L = g.loc[g["sensor"] == "L30", cols]
    S = g.loc[g["sensor"] == "S30", cols]
    if L.empty or S.empty:
        return pd.DataFrame(columns=["block_id", "season", "dt_days", "ndvi_l30", "ndvi_s30",
                                     "dndvi", "devi"])
    m = L.merge(S, on=["block_id", "season"], suffixes=("_l", "_s"))
    m["dt_days"] = (m["acq_date_s"] - m["acq_date_l"]).dt.days
    m = m[m["dt_days"].abs() <= max_days].copy()
    m["absdt"] = m["dt_days"].abs()
    m = m.sort_values(["absdt", "granule_id_s"]).drop_duplicates(["block_id", "granule_id_l"])
    m["dndvi"] = m["ndvi_mean_s"] - m["ndvi_mean_l"]
    m["devi"] = m["evi_mean_s"] - m["evi_mean_l"]
    return (m.rename(columns={"ndvi_mean_l": "ndvi_l30", "ndvi_mean_s": "ndvi_s30"})
            [["block_id", "season", "dt_days", "ndvi_l30", "ndvi_s30", "dndvi", "devi"]]
            .sort_values(["season", "block_id"]).reset_index(drop=True))


def sensor_bias(scene_rows: pd.DataFrame, max_days: int = 1,
                min_clear_frac: float = MIN_BLOCK_CLEAR) -> pd.DataFrame:
    """Per season: n_pairs, mean/median dNDVI and dEVI (S30 - L30) over block pairs."""
    p = sensor_pairs(scene_rows, max_days, min_clear_frac)
    if p.empty:
        return pd.DataFrame(columns=["n_pairs", "mean_dndvi", "median_dndvi", "mean_devi",
                                     "median_devi"])
    out = p.groupby("season").agg(
        n_pairs=("dndvi", "size"), mean_dndvi=("dndvi", "mean"), median_dndvi=("dndvi", "median"),
        mean_devi=("devi", "mean"), median_devi=("devi", "median"))
    out.index.name = "year"
    return out


def prior_s30_offsets(scene_rows: pd.DataFrame, years, max_days: int = 1) -> dict:
    """{season: {"ndvi", "evi"}} = median S30 - L30 over pairs of seasons strictly before it
    (0 when there is no prior pair): the optional, causal offset correction."""
    p = sensor_pairs(scene_rows, max_days)
    out = {}
    for y in years:
        q = p[p["season"] < y]
        out[int(y)] = {"ndvi": float(q["dndvi"].median()) if len(q) else 0.0,
                       "evi": float(q["devi"].median()) if len(q) else 0.0}
    return out


def combine(years=None, states=None, bias_correction: bool = False) -> pd.DataFrame:
    """Composites for every cached season (or ``years``) and state whose mask exists."""
    from cornsat import hlssample

    sts = _states_list(states)
    rows = load_scene_rows(years, sts)
    if rows.empty:
        return pd.DataFrame(columns=COMPOSITE_COLS)
    offs = prior_s30_offsets(rows, sorted(rows["season"].unique())) if bias_correction else None
    parts = []
    for y in sorted(rows["season"].unique()):
        have = [s for s in sts if hlssample._mask_path(s, int(y)).exists()
                and (rows["season"].eq(y) & rows["state"].eq(s)).any()]
        if not have:
            continue
        b = season_blocks(int(y), have)
        info = b[["season", "block_id", "state", "tile", "n_mask"]]
        cy = {(s, int(y)): cdl_years_str(s, int(y)) for s in have}
        parts.append(composite(rows[rows["season"] == y], info, cy, offs))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COMPOSITE_COLS)


def write_state(comps: pd.DataFrame) -> dict[str, int]:
    """Write hls_state.parquet (2016+) and hls_state_presample.parquet (2013-2015)."""
    out = {}
    pre = comps[comps["year"] < SAMPLE_FIRST_YEAR]
    smp = comps[comps["year"] >= SAMPLE_FIRST_YEAR]
    for df, path in ((smp, HLS_STATE_PATH), (pre, HLS_PRESAMPLE_PATH)):
        if len(df):
            _write_atomic(df.reset_index(drop=True), path)
            out[path.name] = len(df)
    return out


def build_all(years=None, states=None, workers: int = DEFAULT_WORKERS,
              bias_correction: bool = False, combine_only: bool = False,
              progress: bool = True) -> dict:
    """Ingest ``years`` (default 2013-2026; skipped with ``combine_only``), then composite
    every cached season and write the processed parquet files."""
    years = list(range(FIRST_YEAR, 2027)) if years is None else list(years)
    summaries = []
    if not combine_only:
        for y in years:
            summaries.append(build_year(y, states, workers, progress=progress))
    comps = combine(None, states, bias_correction)
    written = write_state(comps)
    return {"years": summaries, "written": written, "n_composites": len(comps)}

"""Corn-fraction weight rasters on the MODIS sinusoidal grid.

Output
------
``corn_fraction(state, cdl_year)`` is the fraction (0..1) of 30 m USDA CDL pixels
classified as corn (class 1) inside each MODIS pixel (average resampling of a 0/1 corn
layer; pixels outside the state polygon are zeroed at 30 m *before* averaging, so
boundary pixels carry only in-state corn). The grid is ``modis_geobox(state)``: MODIS
sinusoidal CRS (sphere R=6371007.181 m), native 231.65635826395825 m pixels, edges
snapped to the global MODIS grid (origin x=-20015109.354, y=10007554.677), so Phase 3
can do ``odc.stac.load(..., geobox=modis_geobox(state))`` on MOD13Q1 without any
sub-pixel shift.

CDL source per year
-------------------
* 2008-2021: Microsoft Planetary Computer, collection ``usda-cdl`` (STAC items
  ``cropland_{year}_{x}_{y}_90000``; asset ``cropland``; 3000x3000 px tiles of 90 km,
  EPSG:5070, 30 m, uint8 COG-like with overviews, nodata 0, ~2.4 MB each). Only tiles
  intersecting the state polygon are downloaded, held in memory, never stored. (The
  collection's temporal extent ends 2021-12-31; other asset types in it are
  ``frequency`` and ``cultivated`` which are not used.)
* 2022-2025: USDA CropScape (NASS/GMU) ``GetCDLFile?year=YYYY&fips=SS`` which returns
  XML with a ``returnURL`` to a per-state GeoTIFF (EPSG:5070, 30 m, uncompressed
  strips, ~100-250 MB). It is streamed to a temporary file under
  ``data/interim/cornmask/tmp/``, aggregated, and deleted. Only the corn fraction
  raster (small) is kept.

Look-ahead
----------
CDL for year Y is published in early Y+1, so season Y may only use CDL years
Y-3..Y-1 (``mask_years``). CDL national coverage starts in 2008 (season 2009 -> [2008]).
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr
from affine import Affine
from odc.geo import CRS, wh_
from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_coords
from rasterio.features import rasterize
from rasterio.io import MemoryFile
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window

from cornsat.aoi import load_states
from cornsat.config import DATA_INTERIM, STATES

CORNMASK_DIR = DATA_INTERIM / "cornmask"
TMP_DIR = CORNMASK_DIR / "tmp"

# --- MODIS grid ---------------------------------------------------------------------
MODIS_PROJ = "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs"
MODIS_CRS = CRS(MODIS_PROJ)
MODIS_RES = 231.65635826395825  # m (1 / 4800 of a 1111950.5196666666 m tile)
MODIS_ORIGIN_X = -20015109.354
MODIS_ORIGIN_Y = 10007554.677
PIXEL_AREA_M2 = MODIS_RES**2  # sinusoidal is equal-area
M2_PER_ACRE = 4046.8564224

# --- CDL ----------------------------------------------------------------------------
CDL_FIRST_YEAR = 2008
CDL_LAST_YEAR = 2025
PC_LAST_YEAR = 2021  # Planetary Computer usda-cdl covers 2008-2021
CORN_CLASS = 1
CDL_CRS = "EPSG:5070"
CDL_RES = 30.0
CDL_GRID_OFFSET = 15.0  # CDL pixel edges lie at multiples of 30 m plus 15 m
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CROPSCAPE_URL = "https://nassgeodata.gmu.edu/axis2/services/CDLService/GetCDLFile"

# Cumulative network usage (bytes) of this process, for reporting.
DOWNLOAD_STATS = {"bytes": 0}
_stats_lock = threading.Lock()


def _count_bytes(n: int) -> None:
    with _stats_lock:
        DOWNLOAD_STATS["bytes"] += n


def cdl_source(cdl_year: int) -> str:
    """Name of the data source used for a CDL year."""
    if not CDL_FIRST_YEAR <= cdl_year <= CDL_LAST_YEAR:
        raise ValueError(f"CDL year {cdl_year} outside {CDL_FIRST_YEAR}-{CDL_LAST_YEAR}")
    return "planetary_computer:usda-cdl/cropland" if cdl_year <= PC_LAST_YEAR else "cropscape"


def cdl_years() -> list[int]:
    return list(range(CDL_FIRST_YEAR, CDL_LAST_YEAR + 1))


# --- geometry helpers ---------------------------------------------------------------
def _fips(state_abbr: str) -> str:
    return STATES[state_abbr]


def _state_geom_4326(state_abbr: str) -> gpd.GeoDataFrame:
    gdf = load_states(_fips(state_abbr))
    if len(gdf) != 1:
        raise KeyError(state_abbr)
    return gdf


def modis_geobox(state_abbr: str) -> GeoBox:
    """GeoBox in MODIS sinusoidal at native resolution, grid-aligned, covering the state bbox."""
    gdf = _state_geom_4326(state_abbr).to_crs(MODIS_PROJ)
    minx, miny, maxx, maxy = gdf.total_bounds
    res = MODIS_RES
    i0 = math.floor((minx - MODIS_ORIGIN_X) / res)
    i1 = math.ceil((maxx - MODIS_ORIGIN_X) / res)
    j0 = math.floor((MODIS_ORIGIN_Y - maxy) / res)
    j1 = math.ceil((MODIS_ORIGIN_Y - miny) / res)
    x0 = MODIS_ORIGIN_X + i0 * res
    y0 = MODIS_ORIGIN_Y - j0 * res
    aff = Affine(res, 0.0, x0, 0.0, -res, y0)
    return GeoBox(wh_(i1 - i0, j1 - j0), aff, MODIS_CRS)


def state_mask(state_abbr: str, supersample: int = 4) -> xr.DataArray:
    """Fraction (0..1) of each pixel of ``modis_geobox`` that lies inside the state polygon."""
    gb = modis_geobox(state_abbr)
    geom = _state_geom_4326(state_abbr).to_crs(MODIS_PROJ).geometry.iloc[0]
    k = supersample
    h, w = gb.shape
    fine_aff = gb.affine * Affine.scale(1.0 / k)
    fine = rasterize(
        [(geom, 1)], out_shape=(h * k, w * k), transform=fine_aff, fill=0, dtype="uint8"
    )
    frac = fine.reshape(h, k, w, k).mean(axis=(1, 3)).astype("float32")
    return _to_dataarray(frac, gb, "state_mask")


def _to_dataarray(arr: np.ndarray, gb: GeoBox, name: str) -> xr.DataArray:
    coords = xr_coords(gb)
    return xr.DataArray(arr, dims=("y", "x"), coords=coords, name=name)


# --- no look-ahead ------------------------------------------------------------------
def mask_years(season_year: int) -> list[int]:
    """Up to 3 CDL years in [season_year-3, season_year-1] that exist (>= 2008, <= latest)."""
    lo = max(season_year - 3, CDL_FIRST_YEAR)
    hi = min(season_year - 1, CDL_LAST_YEAR)
    return list(range(lo, hi + 1))


# --- 30 m corn layer in EPSG:5070 ---------------------------------------------------
def _grid_30m(geom_5070) -> tuple[Affine, tuple[int, int]]:
    minx, miny, maxx, maxy = geom_5070.bounds
    c, r = CDL_RES, CDL_GRID_OFFSET
    x0 = math.floor((minx - r) / c) * c + r
    x1 = math.ceil((maxx - r) / c) * c + r
    y1 = math.ceil((maxy - r) / c) * c + r
    y0 = math.floor((miny - r) / c) * c + r
    w = int(round((x1 - x0) / c))
    h = int(round((y1 - y0) / c))
    return Affine(c, 0.0, x0, 0.0, -c, y1), (h, w)


def _paste_corn(dst: np.ndarray, dst_aff: Affine, src: np.ndarray, src_aff: Affine) -> None:
    """OR ``src == corn`` into ``dst`` at the position given by the two (aligned) affines."""
    dx = (src_aff.c - dst_aff.c) / CDL_RES
    dy = (dst_aff.f - src_aff.f) / CDL_RES
    if abs(dx - round(dx)) > 1e-3 or abs(dy - round(dy)) > 1e-3:
        raise ValueError("CDL source not aligned to the 30 m working grid")
    col0, row0 = int(round(dx)), int(round(dy))  # src (0,0) in dst coordinates
    sh, sw = src.shape
    dh, dw = dst.shape
    r0, r1 = max(row0, 0), min(row0 + sh, dh)
    c0, c1 = max(col0, 0), min(col0 + sw, dw)
    if r0 >= r1 or c0 >= c1:
        return
    sub = src[r0 - row0 : r1 - row0, c0 - col0 : c1 - col0]
    dst[r0:r1, c0:c1] |= (sub == CORN_CLASS).astype("uint8")


def _http_get(url: str, **kw) -> bytes:
    import requests

    last = None
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=180, **kw)
            r.raise_for_status()
            return r.content
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}: {last}")


def _fill_from_pc(corn: np.ndarray, aff: Affine, state_abbr: str, cdl_year: int, geom_5070) -> int:
    import planetary_computer as pc
    import pystac_client
    from shapely.geometry import box

    gdf = _state_geom_4326(state_abbr)
    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    search = cat.search(
        collections=["usda-cdl"],
        bbox=list(gdf.total_bounds),
        datetime=f"{cdl_year}-01-01/{cdl_year}-12-31",
        limit=200,
    )
    items = [
        it
        for it in search.items()
        if it.properties.get("usda_cdl:type") == "cropland" and "cropland" in it.assets
    ]
    todo = []
    for it in items:
        t = Affine(*it.properties["proj:transform"][:6])
        h, w = it.properties["proj:shape"]
        tile_box = box(t.c, t.f - h * CDL_RES, t.c + w * CDL_RES, t.f)
        if tile_box.intersects(geom_5070):
            todo.append((it, t))
    if not todo:
        raise RuntimeError(f"no usda-cdl cropland tiles for {state_abbr} {cdl_year}")

    def fetch(item_t):
        it, t = item_t
        data = _http_get(it.assets["cropland"].href)
        _count_bytes(len(data))
        with MemoryFile(data) as mf, mf.open() as ds:
            arr = ds.read(1)
        return arr, t

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=8) as ex:
        for arr, t in ex.map(fetch, todo):
            with lock:
                _paste_corn(corn, aff, arr, t)
    return len(todo)


def _fill_from_cropscape(corn: np.ndarray, aff: Affine, state_abbr: str, cdl_year: int) -> int:
    import requests

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    xml = _http_get(CROPSCAPE_URL, params={"year": cdl_year, "fips": _fips(state_abbr)}).decode()
    m = re.search(r"<returnURL>([^<]+)</returnURL>", xml)
    if not m:
        raise RuntimeError(f"CropScape returned no URL for {state_abbr} {cdl_year}: {xml[:300]}")
    url = m.group(1)
    tmp = TMP_DIR / f"CDL_{cdl_year}_{_fips(state_abbr)}.tif"
    try:
        last = None
        for attempt in range(4):
            try:
                n = 0
                with requests.get(url, stream=True, timeout=180) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            f.write(chunk)
                            n += len(chunk)
                _count_bytes(n)
                last = None
                break
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(5 * (attempt + 1))
        if last is not None:
            raise RuntimeError(f"CropScape download failed {url}: {last}")
        with rasterio.open(tmp) as ds:
            if ds.crs.to_epsg() != 5070 or abs(ds.res[0] - CDL_RES) > 1e-6:
                raise ValueError(f"unexpected CropScape grid: {ds.crs} {ds.res}")
            band = 1024
            for r0 in range(0, ds.height, band):
                win = Window(0, r0, ds.width, min(band, ds.height - r0))
                arr = ds.read(1, window=win)
                _paste_corn(corn, aff, arr, ds.window_transform(win))
    finally:
        if tmp.exists():
            tmp.unlink()
    return 1


def _corn_30m(state_abbr: str, cdl_year: int) -> tuple[np.ndarray, Affine, int]:
    """Binary corn layer (uint8) on a 30 m EPSG:5070 grid covering the state, clipped to it."""
    geom = _state_geom_4326(state_abbr).to_crs(CDL_CRS).geometry.iloc[0]
    aff, shape = _grid_30m(geom)
    corn = np.zeros(shape, dtype="uint8")
    if cdl_year <= PC_LAST_YEAR:
        n_src = _fill_from_pc(corn, aff, state_abbr, cdl_year, geom)
    else:
        n_src = _fill_from_cropscape(corn, aff, state_abbr, cdl_year)
    inside = rasterize([(geom, 1)], out_shape=shape, transform=aff, fill=0, dtype="uint8")
    corn &= inside
    return corn, aff, n_src


# --- public API ---------------------------------------------------------------------
def _cache_path(state_abbr: str, cdl_year: int) -> Path:
    return CORNMASK_DIR / f"{state_abbr}_{cdl_year}.nc"


def corn_acres(frac: xr.DataArray) -> float:
    """Total corn area (acres) = sum(fraction x pixel area)."""
    return float(frac.astype("float64").sum()) * PIXEL_AREA_M2 / M2_PER_ACRE


def corn_fraction(state_abbr: str, cdl_year: int, refresh: bool = False) -> xr.DataArray:
    """Corn fraction (0..1) on ``modis_geobox(state_abbr)`` for a CDL year; cached as .nc."""
    if not CDL_FIRST_YEAR <= cdl_year <= CDL_LAST_YEAR:
        raise ValueError(f"CDL year {cdl_year} outside {CDL_FIRST_YEAR}-{CDL_LAST_YEAR}")
    gb = modis_geobox(state_abbr)
    path = _cache_path(state_abbr, cdl_year)
    if path.exists() and not refresh:
        with xr.open_dataarray(path) as da:
            arr = da.values.astype("float32")
            attrs = dict(da.attrs)
        if arr.shape == tuple(gb.shape):
            out = _to_dataarray(arr, gb, "corn_fraction")
            out.attrs.update(attrs)
            return out

    corn, aff, n_src = _corn_30m(state_abbr, cdl_year)
    n_corn_30m = int(corn.sum())
    h, w = gb.shape
    dst = np.zeros((h, w), dtype="float32")
    reproject(
        source=corn,
        destination=dst,
        src_transform=aff,
        src_crs=CDL_CRS,
        dst_transform=gb.affine,
        dst_crs=MODIS_PROJ,
        resampling=Resampling.average,
        num_threads=os.cpu_count() or 1,
        warp_mem_limit=1024,
    )
    np.clip(dst, 0.0, 1.0, out=dst)
    out = _to_dataarray(dst, gb, "corn_fraction")
    out.attrs.update(
        state=state_abbr,
        cdl_year=int(cdl_year),
        source=cdl_source(cdl_year),
        n_source_files=int(n_src),
        corn_class=CORN_CLASS,
        corn_acres_30m=n_corn_30m * CDL_RES**2 / M2_PER_ACRE,
        corn_acres_modis=corn_acres(out),
        long_name="fraction of 30 m CDL pixels classed corn within MODIS pixel",
    )
    CORNMASK_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.nc")
    out.drop_vars("spatial_ref", errors="ignore").to_netcdf(tmp, engine="scipy")
    tmp.replace(path)
    return out


PRESAMPLE_CDL_YEARS = [CDL_FIRST_YEAR]
"""CDL years used to corn-weight *pre-sample* MODIS seasons (2003-2008), which have no
prior CDL (national coverage starts 2008). Those seasons feed only the Phase 5
climatology baseline (normalisation), never a tradable feature; corn location in the
five core states is very stable year to year, so the 2008 mask is a close proxy."""


def season_weight(state_abbr: str, season_year: int, cdl_years=None) -> xr.DataArray:
    """Mean corn fraction over ``mask_years(season_year)`` (no look-ahead), on modis_geobox.

    ``cdl_years`` optionally overrides the CDL years averaged (used only for the
    pre-sample climatology seasons, see PRESAMPLE_CDL_YEARS); default keeps
    ``mask_years``.
    """
    years = mask_years(season_year) if cdl_years is None else [int(y) for y in cdl_years]
    if not years:
        raise ValueError(f"no CDL years available for season {season_year}")
    fracs = [corn_fraction(state_abbr, y) for y in years]
    w = sum(f.values.astype("float64") for f in fracs) / len(fracs)
    out = _to_dataarray(w.astype("float32"), modis_geobox(state_abbr), "corn_weight")
    out.attrs.update(
        state=state_abbr,
        season_year=int(season_year),
        cdl_years=years,
        n_years=len(years),
        long_name="corn-fraction weight (mean of prior CDL years)",
    )
    return out


# --- raw CDL class codes in a small window (Phase 8b, HLS block masks) ---------------
# Additive helpers: nothing above depends on them. They return the *raw* CDL class codes
# (not only corn) for an EPSG:5070 bounding box on the native 30 m CDL grid.
SOY_CLASS = 5
_GDAL_HTTP_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "2",
    "GDAL_HTTP_TIMEOUT": "120",
}
_cropscape_url_lock = threading.Lock()
_cropscape_urls: dict[tuple[int, str], str] = {}


def _paste_codes(dst: np.ndarray, dst_aff: Affine, ds) -> int:
    """Read the overlap of open dataset ``ds`` (30 m, aligned) with ``dst``; paste non-zero codes.

    Returns the number of pixels read.
    """
    t = ds.transform
    dx = (dst_aff.c - t.c) / CDL_RES
    dy = (t.f - dst_aff.f) / CDL_RES
    if abs(dx - round(dx)) > 1e-3 or abs(dy - round(dy)) > 1e-3 or abs(t.a - CDL_RES) > 1e-6:
        raise ValueError("CDL source not aligned to the 30 m working grid")
    col0, row0 = int(round(dx)), int(round(dy))  # dst (0,0) in src pixel coordinates
    dh, dw = dst.shape
    c0, c1 = max(col0, 0), min(col0 + dw, ds.width)
    r0, r1 = max(row0, 0), min(row0 + dh, ds.height)
    if r0 >= r1 or c0 >= c1:
        return 0
    arr = ds.read(1, window=Window(c0, r0, c1 - c0, r1 - r0))
    sub = dst[r0 - row0 : r1 - row0, c0 - col0 : c1 - col0]
    np.copyto(sub, arr, where=arr != 0)
    _count_bytes(arr.size)
    return int(arr.size)


def _with_retries(fn, tries: int = 4):
    last = None
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * 2**attempt)
    raise RuntimeError(f"CDL window read failed: {last}")


_pc_cdl_items_cache: dict[int, list] = {}
_pc_cdl_items_lock = threading.Lock()
# lon/lat box around the core and extra states; one STAC search per CDL year
_PC_CDL_SEARCH_BBOX = [-104.2, 35.9, -80.4, 49.5]


def _pc_cdl_items(cdl_year: int) -> list:
    """usda-cdl cropland items (unsigned) for one year, searched once and cached.

    One search per block window got the Planetary Computer STAC API to rate-limit the
    mask build, so the year's item list is fetched once and filtered locally."""
    import pystac_client

    with _pc_cdl_items_lock:
        if cdl_year in _pc_cdl_items_cache:
            return _pc_cdl_items_cache[cdl_year]

        def search():
            cat = pystac_client.Client.open(STAC_URL)
            return list(
                cat.search(
                    collections=["usda-cdl"],
                    bbox=_PC_CDL_SEARCH_BBOX,
                    datetime=f"{cdl_year}-01-01/{cdl_year}-12-31",
                    limit=500,
                ).items()
            )

        items = [
            it
            for it in _with_retries(search, tries=6)
            if it.properties.get("usda_cdl:type") == "cropland" and "cropland" in it.assets
        ]
        _pc_cdl_items_cache[cdl_year] = items
        return items


def _pc_cdl_window(dst: np.ndarray, aff: Affine, cdl_year: int) -> int:
    import planetary_computer as pc
    from shapely.geometry import box

    h, w = dst.shape
    bounds = (aff.c, aff.f - h * CDL_RES, aff.c + w * CDL_RES, aff.f)
    items = _pc_cdl_items(cdl_year)
    win = box(*bounds)
    n = 0
    for it in items:
        t = Affine(*it.properties["proj:transform"][:6])
        th, tw = it.properties["proj:shape"]
        if not box(t.c, t.f - th * CDL_RES, t.c + tw * CDL_RES, t.f).intersects(win):
            continue

        def read(href=it.assets["cropland"].href):
            # sign at read time: SAS tokens expire, and the item list is cached per year
            with rasterio.Env(**_GDAL_HTTP_ENV), rasterio.open(pc.sign(href)) as ds:
                return _paste_codes(dst, aff, ds)

        n += _with_retries(read)
    if n == 0:
        raise RuntimeError(f"no usda-cdl cropland data for {bounds} in {cdl_year}")
    return n


def _cropscape_state_url(cdl_year: int, fips: str) -> str:
    """URL of CropScape's cached per-state CDL GeoTIFF (supports HTTP range reads).

    (``GetCDLFile`` also accepts ``bbox=``, but that clips on the fly and returned HTTP 502
    after ~180 s when tested; the per-state file is pre-built and answers in < 1 s.)
    """
    key = (int(cdl_year), fips)
    with _cropscape_url_lock:
        if key in _cropscape_urls:
            return _cropscape_urls[key]
    xml = _http_get(CROPSCAPE_URL, params={"year": cdl_year, "fips": fips}).decode()
    m = re.search(r"<returnURL>([^<]+)</returnURL>", xml)
    if not m:
        raise RuntimeError(f"CropScape returned no URL for {fips} {cdl_year}: {xml[:300]}")
    with _cropscape_url_lock:
        _cropscape_urls[key] = m.group(1)
    return m.group(1)


def _state_of_bounds(bounds_5070) -> str:
    from shapely.geometry import box

    c = box(*bounds_5070).centroid
    for st in STATES:
        if _state_geom_4326(st).to_crs(CDL_CRS).geometry.iloc[0].contains(c):
            return st
    raise ValueError(f"bounds {bounds_5070} are not inside any of {list(STATES)}")


def _cropscape_window(dst: np.ndarray, aff: Affine, cdl_year: int, state_abbr: str) -> int:
    url = _cropscape_state_url(cdl_year, _fips(state_abbr))

    def read():
        with rasterio.Env(**_GDAL_HTTP_ENV), rasterio.open(url) as ds:
            return _paste_codes(dst, aff, ds)

    n = _with_retries(read)
    if n == 0:
        raise RuntimeError(f"CropScape {state_abbr} {cdl_year} does not cover the window")
    return n


def cdl_codes_window(
    cdl_year: int, bounds_5070, state_abbr: str | None = None
) -> tuple[np.ndarray, Affine]:
    """Raw CDL class codes (uint8, 0 = no data) covering an EPSG:5070 bounding box.

    The returned array sits on the native 30 m CDL grid (edges at multiples of 30 m plus
    15 m), snapped outward to cover ``bounds_5070 = (minx, miny, maxx, maxy)``. Only a
    window is read: Planetary Computer ``usda-cdl`` COGs for 2008-2021 (mosaicked if the
    box straddles tiles), CropScape's cached per-state GeoTIFF (HTTP range reads) for
    2022+. CropScape files are per state, so ``state_abbr`` picks the file (default: the
    core state containing the box centre); pixels outside that state are 0.
    """
    from shapely.geometry import box

    if not CDL_FIRST_YEAR <= cdl_year <= CDL_LAST_YEAR:
        raise ValueError(f"CDL year {cdl_year} outside {CDL_FIRST_YEAR}-{CDL_LAST_YEAR}")
    aff, shape = _grid_30m(box(*bounds_5070))
    codes = np.zeros(shape, dtype="uint8")
    if cdl_year <= PC_LAST_YEAR:
        _pc_cdl_window(codes, aff, cdl_year)
    else:
        st = state_abbr or _state_of_bounds(bounds_5070)
        _cropscape_window(codes, aff, cdl_year, st)
    return codes, aff

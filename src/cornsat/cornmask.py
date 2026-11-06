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


def season_weight(state_abbr: str, season_year: int) -> xr.DataArray:
    """Mean corn fraction over ``mask_years(season_year)`` (no look-ahead), on modis_geobox."""
    years = mask_years(season_year)
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

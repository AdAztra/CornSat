"""Area of interest: US state and county boundaries (Census cartographic boundary files).

Files are downloaded once from the Census Bureau and cached under
``data/interim/boundaries/``. Geometries are returned in EPSG:4326.
"""

from __future__ import annotations

import zipfile
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import requests

from cornsat.config import DATA_INTERIM

BOUNDARY_DIR = DATA_INTERIM / "boundaries"
CENSUS_BASE = "https://www2.census.gov/geo/tiger/GENZ2023/shp"
STATE_ZIP = "cb_2023_us_state_20m.zip"
COUNTY_ZIP = "cb_2023_us_county_20m.zip"


def _fetch(zip_name: str) -> Path:
    """Download (once) and unzip a Census boundary file; return the .shp path."""
    BOUNDARY_DIR.mkdir(parents=True, exist_ok=True)
    zpath = BOUNDARY_DIR / zip_name
    shp = BOUNDARY_DIR / zip_name.replace(".zip", "") / zip_name.replace(".zip", ".shp")
    if shp.exists():
        return shp
    if not zpath.exists():
        resp = requests.get(f"{CENSUS_BASE}/{zip_name}", timeout=120)
        resp.raise_for_status()
        tmp = zpath.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.rename(zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(shp.parent)
    return shp


@lru_cache(maxsize=1)
def _states() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_fetch(STATE_ZIP))
    return gdf.to_crs(4326)


@lru_cache(maxsize=1)
def _counties() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_fetch(COUNTY_ZIP))
    return gdf.to_crs(4326)


def load_states(fips: str | list[str] | None = None) -> gpd.GeoDataFrame:
    """US state polygons (EPSG:4326). ``fips`` is a state FIPS code or list of codes."""
    gdf = _states()
    if fips is not None:
        codes = [fips] if isinstance(fips, str) else list(fips)
        gdf = gdf[gdf["STATEFP"].isin(codes)]
    return gdf.reset_index(drop=True).copy()


def load_counties(state_fips: str | list[str] | None = None) -> gpd.GeoDataFrame:
    """US county polygons (EPSG:4326), optionally restricted to state FIPS code(s)."""
    gdf = _counties()
    if state_fips is not None:
        codes = [state_fips] if isinstance(state_fips, str) else list(state_fips)
        gdf = gdf[gdf["STATEFP"].isin(codes)]
    return gdf.reset_index(drop=True).copy()

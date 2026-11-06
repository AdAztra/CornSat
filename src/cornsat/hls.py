"""NASA HLS v2.0 (HLSL30 + HLSS30) access layer: token-safe COG reads over HTTPS.

Search goes through CMR STAC (LPCLOUD). Reads use GDAL /vsicurl with a bearer
token. The token is never logged or included in error messages.
"""

from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import rasterio
import requests
from dotenv import load_dotenv
from rasterio.windows import Window

from cornsat.config import ROOT

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
        m = re.search(r"\.(B\d\dA?|Fmask)\.tif$", u)
        if m:
            assets[m.group(1)] = _A(u)
    props = {"datetime": g["acquisition"], "eo:cloud_cover": g.get("cloud_cover")}
    return parse_item(_Fake(g["id"], props, assets))


def cmr_granules(short_name: str, start, end, bbox=None, tile: str | None = None, page_size: int = 2000):
    """CMR granule search (UMM-JSON) -> dicts: id, acquisition, production, revision, cloud_cover, urls."""
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

"""Probe NASA HLS v2.0 access: auth, COG layout, history, latency, throughput, PC cross-check.

Prints a summary and writes data/interim/hls/probe.json. Total traffic is a few hundred MB.
Never prints the Earthdata token.
"""

from __future__ import annotations

import json
import statistics as st
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import rasterio
import requests
from rasterio.windows import Window

from cornsat import hls
from cornsat.config import DATA_INTERIM

TILE = "T15TVG"
IOWA_BBOX = [-94.0, 41.8, -93.5, 42.2]
OUT = DATA_INTERIM / "hls"


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def parse_dt(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def layout(item):
    out = {}
    with hls.gdal_env():
        for key in ("red", "fmask"):
            with hls.open_cog(item.assets[key]) as s:
                out[key] = dict(
                    shape=list(s.shape), dtype=s.dtypes[0], block=list(s.block_shapes[0]),
                    overviews=s.overviews(1), crs=str(s.crs), nodata=s.nodata,
                )
    return out


def read_blocks(item, keys=("blue", "red", "nir", "fmask"), n=3, y0=1024, x0=1024):
    """Read n*n 256x256 blocks from each asset; returns bytes-equivalent pixel count."""
    npx = 0
    with hls.gdal_env():
        for k in keys:
            with hls.open_cog(item.assets[k]) as s:
                for i in range(n):
                    for j in range(n):
                        a = s.read(1, window=Window(x0 + 256 * j, y0 + 256 * i, 256, 256))
                        npx += a.size
    return npx


def throughput(items, workers):
    """One task per (granule, asset) reading 9 blocks. Returns rate and errors."""
    tasks = [(it, k) for it in items for k in ("blue", "red", "nir", "fmask")]
    errs, n429 = Counter(), 0

    def run(t):
        it, k = t
        try:
            with hls.gdal_env():
                with hls.open_cog(it.assets[k]) as s:
                    for i in range(3):
                        for j in range(3):
                            s.read(1, window=Window(1024 + 256 * j, 1024 + 256 * i, 256, 256))
            return None
        except Exception as e:  # noqa: BLE001
            return type(e).__name__ + ":" + str(e)[:60]

    before = hls._STATS["http_429"]
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(run, tasks))
    dt = time.time() - t0
    for r in res:
        if r:
            errs[r] += 1
    return dict(
        workers=workers, granules=len(items), tasks=len(tasks), seconds=round(dt, 1),
        granule_assets_per_sec=round(len(tasks) / dt, 2),
        errors=dict(errs), http_429=hls._STATS["http_429"] - before,
    )


def latency(label, short, bbox, start, end):
    g = hls.cmr_granules(short, start, end, bbox=bbox)
    prod, rev = [], []
    reproc = 0
    for x in g:
        if not x["acquisition"] or not x["production"]:
            continue
        a = parse_dt(x["acquisition"])
        prod.append((parse_dt(x["production"]) - a).total_seconds() / 86400)
        rev.append((parse_dt(x["revision"]) - a).total_seconds() / 86400)
        reproc += (x["revision_id"] or 1) > 1
    return dict(
        label=label, n=len(prod), reprocessed_rev_gt1=reproc,
        production_lag_days=dict(median=pct(prod, 50), p90=pct(prod, 90), p95=pct(prod, 95), max=max(prod, default=None)),
        revision_lag_days=dict(median=pct(rev, 50), p90=pct(rev, 90), p95=pct(rev, 95)),
    )


def planetary_check(item_id, tile_item):
    r = requests.get(
        "https://planetarycomputer.microsoft.com/api/stac/v1/search",
        params={"collections": "hls2-s30", "ids": item_id}, timeout=60,
    )
    feats = r.json().get("features", [])
    if not feats:
        return dict(item=item_id, found=False)
    assets = feats[0]["assets"]
    tk = requests.get("https://planetarycomputer.microsoft.com/api/sas/v1/token/hls2euwest/hls2", timeout=60).json()["token"]
    res = dict(item=item_id, found=True, pc_asset_keys=sorted(assets)[:20])
    for ours, pc in (("red", "B04"), ("fmask", "Fmask")):
        w = Window(1024, 1024, 256, 256)
        a = hls.read_window(tile_item.assets[ours], w)
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
            with rasterio.open(assets[pc]["href"] + "?" + tk) as s:
                b = s.read(1, window=w)
        res[ours] = dict(identical=bool(np.array_equal(a, b)), max_abs_diff=int(np.abs(a.astype("i4") - b.astype("i4")).max()),
                         valid_frac=float((a != hls.NODATA).mean()) if ours == "red" else None)
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = {}

    # 1. collections / auth
    from pystac_client import Client

    cols = [c.id for c in Client.open(hls.STAC_URL).get_collections() if c.id.startswith("HLS")]
    R["collections"] = cols
    l30 = hls.search(TILE, "2024-06-01", "2024-09-15", collections=("HLSL30_2.0",))
    s30 = hls.search(TILE, "2024-06-01", "2024-09-15", collections=("HLSS30_2.0",))
    t0 = time.time()
    a = hls.read_window(l30[0].assets["red"], Window(1024, 1024, 256, 256))
    R["auth"] = dict(method="requests Bearer -> 303 pre-signed CloudFront URL -> GDAL /vsicurl",
                     ok=bool(a.shape == (256, 256)), read_s=round(time.time() - t0, 2),
                     sample_valid_frac=float((a != hls.NODATA).mean()))
    print("1.", R["collections"], R["auth"])

    # 2. layout
    R["layout"] = {"L30": layout(l30[0]), "S30": layout(s30[0])}
    print("2.", json.dumps(R["layout"]))

    # 3. history
    early = {}
    for short in ("HLSL30", "HLSS30"):
        g = hls.cmr_granules(short, "2012-01-01", "2016-12-31", bbox=IOWA_BBOX, page_size=2000)
        g = sorted(g, key=lambda x: x["acquisition"])
        early[short] = dict(earliest=g[0]["acquisition"] if g else None,
                            AprOct_counts_by_year={
                                y: sum(1 for x in g if x["acquisition"][:4] == str(y) and 4 <= int(x["acquisition"][5:7]) <= 10)
                                for y in (2013, 2014, 2015, 2016)})
    counts = {}
    for y in range(2013, 2026):
        row = {}
        for sensor, short in (("L30", "HLSL30"), ("S30", "HLSS30")):
            row[sensor] = len(hls.cmr_granules(short, f"{y}-06-01", f"{y}-09-15", tile=TILE))
        counts[y] = row
    R["history"] = dict(earliest_iowa_bbox=early, tile=TILE, jun1_sep15_counts=counts)
    print("3.", json.dumps(R["history"]))

    # 4. latency
    R["latency"] = [
        latency("L30 2025-26", "HLSL30", IOWA_BBOX, "2025-01-01", "2026-11-05"),
        latency("S30 2025-26", "HLSS30", IOWA_BBOX, "2025-01-01", "2026-11-05"),
        latency("L30 2019 (historical)", "HLSL30", IOWA_BBOX, "2019-06-01", "2019-09-15"),
        latency("S30 2019 (historical)", "HLSS30", IOWA_BBOX, "2019-06-01", "2019-09-15"),
    ]
    for x in R["latency"]:
        print("4.", json.dumps(x))

    # 5. throughput
    pool = hls.search(TILE, "2023-05-01", "2024-09-30")
    pool = [i for i in pool if i.assets.keys() >= {"blue", "red", "nir", "fmask"}]
    t0 = time.time()
    npx = read_blocks(pool[0])
    seq = time.time() - t0
    R["throughput"] = dict(single_granule_36_blocks_s=round(seq, 2), pixels=npx, pool=len(pool), runs=[])
    for k, w in enumerate((4, 8, 16, 32)):
        chunk = pool[1 + 20 * k: 21 + 20 * k] if k < 3 else pool[61:81]
        R["throughput"]["runs"].append(throughput(chunk, w))
        print("5.", json.dumps(R["throughput"]["runs"][-1]))
    print("5. single granule 4 assets x 9 blocks:", R["throughput"]["single_granule_36_blocks_s"], "s")

    # 6. Planetary Computer cross-check (2024 S30)
    cand = [i for i in s30 if i.assets.keys() >= {"red", "fmask"}]
    R["planetary_computer"] = planetary_check(cand[len(cand) // 2].id, cand[len(cand) // 2])
    print("6.", json.dumps(R["planetary_computer"]))

    (OUT / "probe.json").write_text(json.dumps(R, indent=1, default=str))
    print("wrote", OUT / "probe.json")


if __name__ == "__main__":
    main()

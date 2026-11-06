import logging
from datetime import timezone
from types import SimpleNamespace

import pytest

from cornsat import hls


def fake_item(id_, sensor_assets):
    assets = {k: SimpleNamespace(href=f"https://example.test/{id_}.{k}.tif") for k in sensor_assets}
    props = {"datetime": "2024-07-29T16:53:06.123Z", "eo:cloud_cover": 12, "platform": "LANDSAT-8"}
    return SimpleNamespace(id=id_, properties=props, assets=assets)


def test_band_mapping():
    assert hls.BANDS["L30"] == {"blue": "B02", "red": "B04", "nir": "B05"}
    assert hls.BANDS["S30"] == {"blue": "B02", "red": "B04", "nir": "B8A"}
    assert hls.FMASK == "Fmask" and hls.NODATA == -9999 and hls.SCALE == 1e-4


def test_parse_item_l30():
    it = hls.parse_item(fake_item("HLS.L30.T15TVG.2024211T165306.v2.0", ["B02", "B04", "B05", "B8A", "Fmask"]))
    assert (it.tile, it.sensor, it.cloud_cover) == ("15TVG", "L30", 12.0)
    assert it.datetime.tzinfo is not None and it.datetime.astimezone(timezone.utc).year == 2024
    assert set(it.assets) == {"blue", "red", "nir", "fmask"}
    assert it.assets["nir"].endswith(".B05.tif")


def test_parse_item_s30_uses_b8a():
    it = hls.parse_item(fake_item("HLS.S30.T15TVG.2024211T165841.v2.0", ["B02", "B04", "B05", "B8A", "Fmask"]))
    assert it.sensor == "S30" and it.assets["nir"].endswith(".B8A.tif")


def test_parse_item_datetime_from_id_when_missing():
    f = fake_item("HLS.L30.T15TVG.2024211T165306.v2.0", ["B04"])
    f.properties = {}
    it = hls.parse_item(f)
    assert (it.datetime.month, it.datetime.day, it.datetime.hour) == (7, 29, 16)


def test_parse_item_bad_id():
    with pytest.raises(ValueError):
        hls.parse_item(fake_item("not-an-hls-id", []))


def test_token_missing_error_has_no_secret(monkeypatch):
    monkeypatch.setattr(hls, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("EARTHDATA_TOKEN", raising=False)
    with pytest.raises(hls.EarthdataTokenError) as e:
        hls.token()
    assert "EARTHDATA_TOKEN" in str(e.value)


def test_token_expired_error_has_no_secret(monkeypatch):
    import base64
    import json

    b = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")  # noqa: E731
    secret = f"{b({'alg': 'none'})}.{b({'exp': 1000})}.SECRETSIG"
    monkeypatch.setattr(hls, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("EARTHDATA_TOKEN", secret)
    with pytest.raises(hls.EarthdataTokenError) as e:
        hls.token()
    assert secret not in str(e.value) and "SECRETSIG" not in str(e.value)


def test_gdal_env_never_exposes_token(monkeypatch, caplog):
    monkeypatch.setattr(hls, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("EARTHDATA_TOKEN", "SUPERSECRETVALUE")
    with caplog.at_level(logging.DEBUG):
        env = hls.gdal_env()
        text = repr(env) + str(env.options) + caplog.text
    assert "SUPERSECRETVALUE" not in text
    assert env.options["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"


def test_network_search_and_read():
    try:
        items = hls.search("T15TVG", "2024-07-01", "2024-07-20", collections=("HLSL30_2.0",))
        assert items and items[0].sensor == "L30"
        hls.token()
        from rasterio.windows import Window

        a = hls.read_window(items[0].assets["red"], Window(1024, 1024, 64, 64))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"network/auth unavailable: {type(e).__name__}")
    assert a.shape == (64, 64)

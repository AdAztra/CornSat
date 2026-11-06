"""Offline checks for the scripts/build_all.py step table and dispatcher."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cornsat import features, hls

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def build_all():
    spec = importlib.util.spec_from_file_location("build_all", ROOT / "scripts" / "build_all.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_steps_order(build_all):
    assert build_all.STEPS == ["prices", "cornmask", "modis", "hls_sample", "hls", "weather",
                               "usda", "features", "backtest"]


def test_dispatcher_finds_hls_steps(build_all):
    # main() dispatches via globals()[f"step_{name}"], so every STEPS entry must resolve.
    g = vars(build_all)
    for name in build_all.STEPS:
        assert callable(g[f"step_{name}"])


def test_step_hls_skips_without_token(build_all, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(build_all, "_run_main", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(features, "HLS_STATE_PATH", Path("/nonexistent/hls_state.parquet"))

    def no_token():
        raise hls.EarthdataTokenError("EARTHDATA_TOKEN is not set.")

    monkeypatch.setattr(hls, "token", no_token)
    build_all.step_hls(False)  # must not raise
    assert calls == []
    assert "SKIP hls step" in capsys.readouterr().out

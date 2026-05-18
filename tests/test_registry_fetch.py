"""Tests for registry_fetch module."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ppmlx.registry_fetch import (
    _extract_lab,
    _extract_modalities,
    _extract_params_b,
    _extract_precision,
    _extract_updated_at,
    _repo_id_to_alias,
    cache_status_text,
    is_cache_stale,
    maybe_refresh,
)


# ── Alias generation ─────────────────────────────────────────────────


@pytest.mark.parametrize("repo_id, expected", [
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "qwen3.5:9b"),
    ("mlx-community/Qwen3.5-0.8B-8bit", "qwen3.5:0.8b-8bit"),
    ("mlx-community/GLM-4.7-Flash-4bit", "glm-4.7-flash"),
    ("mlx-community/GLM-4.7-Flash-8bit", "glm-4.7-flash-8bit"),
    ("mlx-community/Kimi-K2.5", "kimi-k2.5"),
    ("mlx-community/gemma-4-26b-a4b-it-4bit", "gemma-4:26b-a4b"),
    ("mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", "devstral-small-2:24b"),
    ("mlx-community/Qwen3.5-35B-A3B-4bit", "qwen3.5:35b-a3b"),
    ("mlx-community/parakeet-tdt-0.6b-v3", "parakeet-tdt:0.6b-v3"),
])
def test_repo_id_to_alias(repo_id: str, expected: str):
    assert _repo_id_to_alias(repo_id) == expected


def test_repo_id_to_alias_no_slash():
    assert _repo_id_to_alias("no-slash") is None


# ── Cache staleness ──────────────────────────────────────────────────


def test_staleness_always(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "cache.json"):
        assert is_cache_stale("always") is True


def test_staleness_never(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "cache.json"):
        assert is_cache_stale("never") is False


def test_staleness_weekly_fresh(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fetched_at": time.time() - 3600}))  # 1 hour ago
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=cache):
        assert is_cache_stale("weekly") is False


def test_staleness_weekly_old(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fetched_at": time.time() - 8 * 86400}))  # 8 days ago
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=cache):
        assert is_cache_stale("weekly") is True


def test_staleness_no_cache_file(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "nope.json"):
        assert is_cache_stale("weekly") is True


def test_staleness_refreshes_old_cache_without_updated_at(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fetched_at": time.time(), "models": {"m": {"repo_id": "x/y"}}}))
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=cache):
        assert is_cache_stale("weekly") is True


def test_cache_status_text_never_without_cache(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "nope.json"):
        assert cache_status_text() == "top downloads refreshed: never"


def test_cache_status_text_includes_date_and_count(tmp_path: Path):
    cache = tmp_path / "cache.json"
    cache.write_text(json.dumps({"fetched_at": 1700000000, "models": {"a": {}, "b": {}}}))
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=cache):
        text = cache_status_text()
    assert "top downloads refreshed: 2023-" in text
    assert "(2 fetched)" in text


def test_fetch_from_hf_uses_top_downloaded_limit():
    from ppmlx.registry_fetch import _fetch_from_hf

    calls = []

    class FakeApi:
        def list_models(self, **kwargs):
            calls.append(kwargs)
            models = []
            for i in range(120):
                m = MagicMock()
                m.id = f"mlx-community/Test-{i + 1}B-4bit"
                m.tags = []
                m.safetensors = {"total": 1024**3}
                m.downloads = 1000 - i
                m.created_at = None
                models.append(m)
            return models

    with patch("huggingface_hub.HfApi", return_value=FakeApi()):
        data = _fetch_from_hf()

    assert calls[0]["author"] == "mlx-community"
    assert calls[0]["sort"] == "downloads"
    assert calls[0]["limit"] >= 100
    assert "lastModified" in calls[0]["expand"]
    assert "createdAt" in calls[0]["expand"]
    assert data is not None
    assert data["source"] == "huggingface-api-downloads"
    assert len(data["models"]) == 100
    assert list(data["models"])[0] == "test:1b"
    assert list(data["models"])[-1] == "test:100b"


# ── maybe_refresh ────────────────────────────────────────────────────


def test_maybe_refresh_never_returns_cache(tmp_path: Path):
    cache = tmp_path / "cache.json"
    data = {"models": {"test:1b": {"repo_id": "mlx-community/test-1B-4bit"}}}
    cache.write_text(json.dumps(data))
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=cache):
        result = maybe_refresh("never")
    assert result is not None
    assert "test:1b" in result["models"]


def test_maybe_refresh_never_no_cache(tmp_path: Path):
    with patch("ppmlx.registry_fetch.get_cache_path", return_value=tmp_path / "nope.json"):
        result = maybe_refresh("never")
    assert result is None


def test_maybe_refresh_fetch_failure_falls_back(tmp_path: Path):
    cache = tmp_path / "cache.json"
    data = {"fetched_at": time.time() - 999999, "models": {"old:1b": {}}}
    cache.write_text(json.dumps(data))
    with (
        patch("ppmlx.registry_fetch.get_cache_path", return_value=cache),
        patch("ppmlx.registry_fetch._fetch_from_hf", return_value=None),
    ):
        result = maybe_refresh("weekly")
    assert result is not None
    assert "old:1b" in result["models"]


# ── Metadata extraction ──────────────────────────────────────────────


def _make_model(repo_id: str, tags: list[str] | None = None) -> MagicMock:
    m = MagicMock()
    m.id = repo_id
    m.tags = tags or []
    return m


def test_extract_lab_known():
    assert _extract_lab(_make_model("mlx-community/Qwen3.5-9B")) == "Alibaba"
    assert _extract_lab(_make_model("mlx-community/Llama-3.1-8B")) == "Meta"
    assert _extract_lab(_make_model("mlx-community/gemma-4-26b")) == "Google"


def test_extract_lab_unknown():
    assert _extract_lab(_make_model("mlx-community/SomeNew-Model")) is None


def test_extract_modalities_text():
    assert _extract_modalities(_make_model("mlx-community/Qwen3-4B")) == "text"


def test_extract_modalities_vision():
    assert _extract_modalities(_make_model("mlx-community/Qwen3-VL-4B")) == "text, vision"


def test_extract_modalities_audio():
    assert _extract_modalities(_make_model("mlx-community/whisper-large")) == "audio, speech"


def test_extract_params_b():
    assert _extract_params_b(_make_model("mlx-community/Qwen3.5-9B-4bit")) == 9.0
    assert _extract_params_b(_make_model("mlx-community/parakeet-0.6b")) == 0.6
    assert _extract_params_b(_make_model("mlx-community/Kimi-K2.5")) is None


def test_extract_params_b_from_safetensors_total():
    model = _make_model("mlx-community/Kimi-K2.5")
    model.safetensors = MagicMock(total=12_345_000_000)
    assert _extract_params_b(model) == 12.3


def test_extract_params_b_from_safetensors_parameters():
    model = _make_model("mlx-community/custom-model")
    model.safetensors = MagicMock(total=None, parameters={"BF16": 1_000_000_000, "U8": 500_000_000})
    assert _extract_params_b(model) == 1.5


@pytest.mark.parametrize("repo_id, expected", [
    ("mlx-community/Qwen3.5-9B-MLX-4bit", "4bit"),
    ("mlx-community/Qwen3.5-0.8B-8bit", "8bit"),
    ("mlx-community/gpt-oss-20b-MXFP4-Q8", "mxfp4-q8"),
    ("mlx-community/HiDream-O1-Image-Dev-mlx-bf16", "bf16"),
    ("mlx-community/Kimi-K2.5", None),
])
def test_extract_precision(repo_id: str, expected: str | None):
    assert _extract_precision(repo_id) == expected


def test_extract_updated_at_prefers_last_modified():
    m = _make_model("mlx-community/Test-1B-4bit")
    m.last_modified = datetime(2026, 5, 18, tzinfo=timezone.utc)
    m.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert _extract_updated_at(m) == "2026-05-18"


# ── Registry load semantics ──────────────────────────────────────────


def test_registry_uses_fetched_cache_without_bundled_merge(monkeypatch):
    import importlib
    import sys
    import types

    if not isinstance(sys.modules.get("ppmlx.registry"), types.ModuleType):
        sys.modules.pop("ppmlx.registry", None)
    registry = importlib.import_module("ppmlx.registry")

    fetched = {
        "version": 1,
        "updated": "2026-05-18",
        "source": "huggingface-api-downloads",
        "models": {
            "top:1b": {"repo_id": "mlx-community/Top-1B-4bit", "downloads": 999},
        },
    }

    monkeypatch.setattr(registry, "_cache", None)
    monkeypatch.setattr("ppmlx.registry_fetch.maybe_refresh", lambda mode: fetched)

    assert registry.registry_aliases() == {"top:1b": "mlx-community/Top-1B-4bit"}
    assert registry.registry_meta()["count"] == 1
    monkeypatch.setattr(registry, "_cache", None)

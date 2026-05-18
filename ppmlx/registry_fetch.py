"""Fetch trending MLX models from HuggingFace and cache locally."""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_FILE = "registry_cache.json"
_HF_AUTHOR = "mlx-community"
_FETCH_LIMIT = 24
_FETCH_TIMEOUT = 8  # seconds

_STALENESS_SECONDS: dict[str, float] = {
    "always": 0,
    "weekly": 7 * 86_400,
    "monthly": 30 * 86_400,
    "never": float("inf"),
}


def get_cache_path() -> Path:
    from ppmlx.config import get_ppmlx_dir
    return get_ppmlx_dir() / _CACHE_FILE


def is_cache_stale(refresh_mode: str) -> bool:
    if refresh_mode == "always":
        return True
    if refresh_mode == "never":
        return False
    cache = get_cache_path()
    if not cache.exists():
        return True
    try:
        data = json.loads(cache.read_text())
        fetched_at = data.get("fetched_at", 0)
        age = time.time() - fetched_at
        return age > _STALENESS_SECONDS.get(refresh_mode, 7 * 86_400)
    except Exception:
        return True


def maybe_refresh(refresh_mode: str) -> dict[str, Any] | None:
    """Return cached data, refreshing first if stale. Returns None on failure."""
    if refresh_mode == "never":
        return _load_cache()
    if is_cache_stale(refresh_mode):
        fresh = _fetch_from_hf()
        if fresh is not None:
            _save_cache(fresh)
            return fresh
    return _load_cache()


def _load_cache() -> dict[str, Any] | None:
    cache = get_cache_path()
    if not cache.exists():
        return None
    try:
        return json.loads(cache.read_text())
    except Exception:
        return None


def cache_status_text() -> str:
    """Return a compact human-readable summary of the dynamic registry cache."""
    data = _load_cache()
    if not data:
        return "last refresh: never"
    fetched_at = data.get("fetched_at")
    if not fetched_at:
        return "last refresh: unknown"
    try:
        dt = datetime.fromtimestamp(float(fetched_at)).astimezone()
        count = len(data.get("models", {}))
        return f"last refresh: {dt.strftime('%Y-%m-%d %H:%M')} ({count} models)"
    except Exception:
        return "last refresh: unknown"


def _save_cache(data: dict[str, Any]) -> None:
    cache = get_cache_path()
    cache.write_text(json.dumps(data, indent=2))


def _fetch_from_hf() -> dict[str, Any] | None:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        models = list(api.list_models(
            author=_HF_AUTHOR,
            sort="trendingScore",
            limit=_FETCH_LIMIT,
            expand=["safetensors", "downloads", "trendingScore"],
            token=False,
        ))
        entries: dict[str, dict[str, Any]] = {}
        for m in models:
            alias = _repo_id_to_alias(m.id)
            if alias is None or alias in entries:
                continue
            size_gb = _estimate_size_gb(m)
            entries[alias] = {
                "repo_id": m.id,
                "params_b": _extract_params_b(m),
                "size_gb": size_gb,
                "type": "dense",
                "lab": _extract_lab(m),
                "modalities": _extract_modalities(m),
                "downloads": m.downloads or 0,
                "created": m.created_at.strftime("%Y-%m-%d") if m.created_at else None,
            }
        return {
            "version": 1,
            "fetched_at": time.time(),
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "source": "huggingface-api",
            "models": entries,
        }
    except Exception as e:
        logger.debug("HF registry fetch failed: %s", e)
        return None


# ── Alias generation ─────────────────────────────────────────────────


def _repo_id_to_alias(repo_id: str) -> str | None:
    """Generate a short alias from an mlx-community repo ID.

    Examples:
        mlx-community/Qwen3.5-9B-MLX-4bit     -> qwen3.5:9b
        mlx-community/GLM-4.7-Flash-4bit       -> glm-4.7-flash
        mlx-community/gpt-oss-20b-MXFP4-Q8     -> gpt-oss:20b-mxfp4-q8
        mlx-community/Qwen3.5-0.8B-8bit        -> qwen3.5:0.8b-8bit
    """
    if "/" not in repo_id:
        return None
    name = repo_id.split("/", 1)[1]
    lower = name.lower()

    # Strip quantization suffix
    quant_match = re.search(r'-(4bit|8bit|3bit|5bit|6bit|bf16|fp16|mxfp4[^-]*)$', lower)
    quant_suffix = ""
    base = lower
    if quant_match:
        quant_str = quant_match.group(1)
        base = lower[:quant_match.start()]
        if quant_str != "4bit":
            quant_suffix = f"-{quant_str}"

    # Strip non-identity suffixes (order matters: dates first, then labels)
    base = re.sub(r'-\d{4}$', '', base)  # date stamps like -2507, -2512
    base = re.sub(r'-(mlx|instruct|chat|hf|gguf|it)$', '', base)

    # Extract param count (e.g. 9b, 0.6b, 120b)
    param_match = re.search(r'-(\d+\.?\d*b)(?:-|$)', base)
    if param_match:
        param_str = param_match.group(1)
        family = base[:param_match.start()]
        remainder = base[param_match.end():]
        if remainder:
            remainder = f"-{remainder.strip('-')}"
        alias = f"{family}:{param_str}{remainder}{quant_suffix}"
    else:
        alias = f"{base}{quant_suffix}"

    alias = re.sub(r'-+', '-', alias).strip('-')
    return alias or None


# ── Metadata extraction heuristics ───────────────────────────────────


_LAB_PATTERNS: dict[str, str] = {
    "Qwen": "Alibaba", "GLM": "Zhipu AI", "gpt-oss": "OpenAI",
    "Llama": "Meta", "Mistral": "Mistral AI", "Devstral": "Mistral AI",
    "Ministral": "Mistral AI", "Gemma": "Google", "gemma": "Google",
    "Phi": "Microsoft", "DeepSeek": "DeepSeek", "deepseek": "DeepSeek",
    "Kimi": "Moonshot AI", "MiniMax": "MiniMax",
    "parakeet": "NVIDIA", "granite": "IBM",
    "LFM": "Liquid AI", "Command": "Cohere", "Jamba": "AI21",
    "SmolLM": "Hugging Face", "Yi": "01.AI", "InternLM": "Shanghai AI Lab",
    "Falcon": "TII", "Codestral": "Mistral AI",
}


def _extract_lab(model: Any) -> str | None:
    name = model.id.split("/", 1)[1] if "/" in model.id else model.id
    for prefix, lab in _LAB_PATTERNS.items():
        if name.startswith(prefix):
            return lab
    return None


def _extract_modalities(model: Any) -> str:
    tags = set(model.tags or [])
    name = model.id.lower()
    if any(x in name for x in ["-vl-", "-vlm", "-vision"]):
        return "text, vision"
    if any(x in name for x in ["whisper", "parakeet", "-asr", "-tts", "-audio"]):
        return "audio, speech"
    if "image-text-to-text" in tags or "visual-question-answering" in tags:
        return "text, vision"
    return "text"


def _extract_params_b(model: Any) -> float | None:
    name = model.id.split("/", 1)[1] if "/" in model.id else model.id
    m = re.search(r'(\d+\.?\d*)[Bb](?:-|$|[A-Z])', name)
    if m:
        return float(m.group(1))
    return None


def _estimate_size_gb(model: Any) -> float | None:
    sf = getattr(model, 'safetensors', None)
    if sf and isinstance(sf, dict):
        total = sf.get('total', 0)
        if total > 0:
            return round(total / (1024**3), 1)
    return None

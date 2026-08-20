"""Deterministic local model family to output-profile selection."""
from __future__ import annotations

from .normalization import NormalizationProfile


def select_normalization_profile(model: str) -> NormalizationProfile | None:
    """Select a profile only when the model identifier names a supported family."""
    if type(model) is not str:
        return None
    normalized = model.lower().replace("_", "-")
    if "grok" in normalized:
        return NormalizationProfile.GROK_OPENAI_CHAT_V1
    if "kimi" in normalized or "moonshot" in normalized:
        return NormalizationProfile.KIMI_K2_V1
    if "deepseek" in normalized:
        return NormalizationProfile.DEEPSEEK_V3_V1
    if "qwen" in normalized:
        return NormalizationProfile.QWEN_JSON_V1
    return None


__all__ = ["select_normalization_profile"]

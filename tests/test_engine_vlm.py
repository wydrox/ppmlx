"""Tests for ppmlx/engine_vlm.py"""
from __future__ import annotations
import base64
import sys
import pytest

from ppmlx.engine_vlm import VisionEngine, get_vision_engine


def test_extract_images_empty():
    engine = VisionEngine()
    messages = [{"role": "user", "content": "Hello"}]
    result = engine._extract_images(messages)
    assert result == []


def test_extract_images_url():
    engine = VisionEngine()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                }
            ],
        }
    ]
    result = engine._extract_images(messages)
    assert len(result) == 1
    assert result[0] == "https://example.com/image.jpg"


def test_extract_images_base64():
    engine = VisionEngine()
    # Create a minimal valid base64 image payload
    fake_bytes = b"\xff\xd8\xff"  # JPEG magic bytes
    b64 = base64.b64encode(fake_bytes).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_uri},
                }
            ],
        }
    ]
    result = engine._extract_images(messages)
    assert len(result) == 1
    # Should be a temp file path (str), not a URL
    assert isinstance(result[0], str)
    assert not result[0].startswith("data:")


def test_generate_calls_vlm(monkeypatch):
    """Mock mlx_vlm.load and mlx_vlm.generate, verify they are called."""
    import ppmlx.engine_vlm as engine_vlm_module

    mock_model = object()
    mock_processor = object()
    load_calls = []
    generate_calls = []

    def mock_load(path):
        load_calls.append(path)
        return mock_model, mock_processor

    def mock_generate(model, processor, prompt, image, max_tokens, temp, verbose):
        generate_calls.append({"prompt": prompt, "image": image})
        return "A cat sitting."

    # Patch mlx_vlm module
    mlx_vlm_mod = sys.modules["mlx_vlm"]
    mlx_vlm_mod.load = mock_load
    mlx_vlm_mod.generate = mock_generate

    engine = VisionEngine()
    messages = [{"role": "user", "content": "What is in this image?"}]

    text, prompt_tokens, completion_tokens = engine.generate(
        "test/model", messages, max_tokens=256
    )

    assert len(load_calls) == 1
    assert len(generate_calls) == 1
    assert text == "A cat sitting."
    assert prompt_tokens > 0
    assert completion_tokens > 0


def test_get_tokenizer_returns_processor_tokenizer():
    """The server can use a vision tokenizer for tool-call parsing."""
    tokenizer = object()
    processor = type("Processor", (), {"tokenizer": tokenizer})()
    mlx_vlm_mod = sys.modules["mlx_vlm"]
    mlx_vlm_mod.load = lambda path: (object(), processor)

    engine = VisionEngine()

    assert engine.get_tokenizer("test/vision-model") is tokenizer


def test_get_tokenizer_uses_processor_as_fallback():
    """A processor without a tokenizer remains a usable parser fallback."""
    processor = object()
    mlx_vlm_mod = sys.modules["mlx_vlm"]
    mlx_vlm_mod.load = lambda path: (object(), processor)

    engine = VisionEngine()

    assert engine.get_tokenizer("test/vision-model") is processor


def test_list_loaded(monkeypatch):
    """After loading a model, list_loaded should contain the repo_id."""
    mock_model = object()
    mock_processor = object()

    mlx_vlm_mod = sys.modules["mlx_vlm"]
    mlx_vlm_mod.load = lambda path: (mock_model, mock_processor)

    engine = VisionEngine()
    engine.load("myorg/myvlm")
    assert "myorg/myvlm" in engine.list_loaded()


def test_unload_all(monkeypatch):
    """After unload_all, list_loaded should be empty."""
    mock_model = object()
    mock_processor = object()

    mlx_vlm_mod = sys.modules["mlx_vlm"]
    mlx_vlm_mod.load = lambda path: (mock_model, mock_processor)

    engine = VisionEngine()
    engine.load("myorg/myvlm2")
    assert len(engine.list_loaded()) == 1

    engine.unload_all()
    assert engine.list_loaded() == []

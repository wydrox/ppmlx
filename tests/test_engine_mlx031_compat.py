"""Regression tests for mlx-lm 0.31.3 compatibility (seed parameter removed).

mlx-lm 0.31.x removed the ``seed`` keyword from ``generate`` /
``generate_step``.  TextEngine must not pass ``seed=`` to mlx_lm; it must
instead seed ``mlx.core.random`` directly, and generation must still work.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


def _make_fake_mlx_lm_0_31(generate_return="tok tok"):
    """Fake mlx_lm whose generate() rejects ``seed`` like the real 0.31.3 API."""
    fake = types.ModuleType("mlx_lm")

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "prompt"
    mock_tokenizer.encode.side_effect = lambda s: list(range(len(s.split())))

    def _generate(model, tokenizer, prompt=None, **kwargs):
        # Mimic mlx-lm 0.31.3: no ``seed`` parameter accepted.
        if "seed" in kwargs:
            raise TypeError(
                "generate_step() got an unexpected keyword argument 'seed'"
            )
        if "sampler" in kwargs and not callable(kwargs["sampler"]):
            raise TypeError("sampler must be callable")
        return generate_return

    fake.generate = _generate
    fake.load = MagicMock(return_value=(mock_model, mock_tokenizer))
    return fake, mock_model, mock_tokenizer


def _install_fake_mlx_core(monkeypatch):
    """Install a fake mlx.core with a seedable random module; return the mock."""
    seed_mock = MagicMock()
    random_mod = types.ModuleType("mlx.core.random")
    random_mod.seed = seed_mock
    core_mod = types.ModuleType("mlx.core")
    core_mod.random = random_mod
    monkeypatch.setitem(sys.modules, "mlx", types.ModuleType("mlx"))
    monkeypatch.setitem(sys.modules, "mlx.core", core_mod)
    monkeypatch.setitem(sys.modules, "mlx.core.random", random_mod)
    return seed_mock


@pytest.fixture(autouse=True)
def _reset(tmp_path, monkeypatch):
    monkeypatch.setattr("ppmlx.engine._resolve_model_path", lambda _: str(tmp_path / "m"))
    from ppmlx.engine import reset_engine

    reset_engine()
    yield
    reset_engine()


def test_generate_works_with_mlx_lm_0_31_no_seed_kwarg(monkeypatch):
    """TextEngine.generate must produce tokens when mlx_lm.generate has no seed param."""
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm_0_31()
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    from ppmlx.engine import TextEngine

    engine = TextEngine()
    result = engine.generate(
        "some/model",
        [{"role": "user", "content": "hi"}],
        temperature=0.7,
        top_p=0.9,
        max_tokens=16,
        seed=42,
    )
    assert result.text == "tok tok"
    assert result.completion_tokens > 0


def test_generate_seeds_mlx_rng_instead_of_passing_seed(monkeypatch):
    """A provided seed must go to mlx.core.random.seed, not into kwargs."""
    fake, _, _ = _make_fake_mlx_lm_0_31()
    calls: dict = {}

    def _spy_generate(model, tokenizer, prompt=None, **kwargs):
        calls["kwargs"] = kwargs
        return "out"

    fake.generate = _spy_generate

    seed_mock = _install_fake_mlx_core(monkeypatch)
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    from ppmlx.engine import TextEngine

    engine = TextEngine()
    engine.generate(
        "some/model",
        [{"role": "user", "content": "hi"}],
        max_tokens=8,
        seed=1234,
    )
    assert "seed" not in calls["kwargs"]
    seed_mock.assert_called_once_with(1234)

    # No seed → RNG untouched by us during this call path.
    seed_mock.reset_mock()
    calls.clear()
    engine.generate(
        "some/model",
        [{"role": "user", "content": "hi again"}],
        max_tokens=8,
    )
    assert "seed" not in calls["kwargs"]
    seed_mock.assert_not_called()


def test_stream_generate_works_with_mlx_lm_0_31_no_seed_kwarg(monkeypatch):
    """stream_generate must also avoid passing seed to mlx_lm."""
    fake, _, _ = _make_fake_mlx_lm_0_31()

    def _stream(model, tokenizer, prompt=None, **kwargs):
        if "seed" in kwargs:
            raise TypeError(
                "generate_step() got an unexpected keyword argument 'seed'"
            )
        for chunk in ("a", "b"):
            obj = MagicMock()
            obj.text = chunk
            yield obj

    fake.stream_generate = _stream

    monkeypatch.setitem(sys.modules, "mlx_lm", fake)

    from ppmlx.engine import TextEngine

    engine = TextEngine()
    chunks = list(
        engine.stream_generate(
            "some/model",
            [{"role": "user", "content": "hi"}],
            max_tokens=8,
            seed=7,
        )
    )
    assert chunks == ["a", "b"]


def test_profile_runner_records_generation_error_detail():
    """The tool-profile runner must surface exception details, not bare 'generation_failed'."""
    from ppmlx.local_runtime.profile_evaluation import (
        ExpectedToolCall,
        ToolEvaluationCase,
        ToolEvaluationCaseSet,
    )
    from ppmlx.local_runtime.profile_runner import (
        GenerationSettings,
        _run_once,
    )

    case = ToolEvaluationCase(
        case_id="c1",
        messages=({"role": "user", "content": "hi"},),
        tools=(),
        expected_calls=(ExpectedToolCall(name="f", arguments={}),),
    )
    case_set = ToolEvaluationCaseSet(
        schema_version="1",
        case_set_version="1",
        cases=(case,),
    )

    def _boom(**kwargs):
        raise TypeError("generate_step() got an unexpected keyword argument 'seed'")

    run = _run_once(
        run_index=1,
        seed=17,
        case_set=case_set,
        model_path="m",
        normalization_profile=None,
        repair_policy=None,
        settings=GenerationSettings(),
        generate=_boom,
    )
    code = run.cases[0].effective.error_code
    assert code is not None
    assert code.startswith("generation_failed:")
    assert "TypeError" in code
    assert "seed" in code

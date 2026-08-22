"""Tests for the Gemma 4 (gemma4-v1) and LFM2.5 (lfm25-v1) profiles."""
from __future__ import annotations

import json

import pytest

from ppmlx.local_runtime.normalization import (
    NormalizationProfile,
    ToolNormalizationError,
    ToolOutputLimits,
    normalize_tool_output,
)


def _roundtrip_ok(call: object) -> bool:
    return json.loads(call.arguments_raw) == call.arguments_json  # type: ignore[attr-defined]


def test_gemma4_keeps_decoded_arguments_and_leading_text() -> None:
    text = (
        "Let me check.\n"
        '<|tool_call>call:read_file{path:"src/app.py",line:42}'
    )
    output = normalize_tool_output(text, profile=NormalizationProfile.GEMMA4_V1)

    assert output.profile is NormalizationProfile.GEMMA4_V1
    assert output.remaining_text == "Let me check.\n"
    assert len(output.tool_calls) == 1
    call = output.tool_calls[0]
    assert call.index == 0
    assert call.name == "read_file"
    assert call.arguments_json == {"path": "src/app.py", "line": 42}
    assert json_loads_roundtrip_ok(call)
    assert call.call_id is None
    assert call.repair is None


def json_loads_roundtrip_ok(call: object) -> bool:
    return json.loads(call.arguments_raw) == call.arguments_json  # type: ignore[attr-defined]


def test_gemma4_accepts_empty_and_nested_arguments() -> None:
    text = (
        "<|tool_call>call:ping{}"
        "<|tool_call>call:search{query:\"a,b\",filters:{tag:[\"x\",1],deep:{k:true}}}"
    )
    output = normalize_tool_output(text, profile="gemma4-v1")

    assert [(call.name, call.arguments_json) for call in output.tool_calls] == [
        ("ping", {}),
        (
            "search",
            {"query": "a,b", "filters": {"tag": ["x", 1], "deep": {"k": True}}},
        ),
    ]


def test_gemma4_rejects_text_after_calls() -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            '<|tool_call>call:read{path:"a"} trailing prose',
            profile=NormalizationProfile.GEMMA4_V1,
        )

    assert captured.value.code == "text_after_tool_section"


def test_lfm25_parses_single_python_style_call() -> None:
    text = (
        "Checking weather."
        '<|tool_call_start|>[get_weather(city="Warsaw", units="celsius")]<|tool_call_end|>'
    )
    output = normalize_tool_output(text, profile=NormalizationProfile.LFM25_V1)

    assert output.profile is NormalizationProfile.LFM25_V1
    assert output.remaining_text == "Checking weather."
    assert len(output.tool_calls) == 1
    call = output.tool_calls[0]
    assert call.index == 0
    assert call.name == "get_weather"
    assert call.arguments_json == {"city": "Warsaw", "units": "celsius"}
    assert _roundtrip_ok(call)


def test_lfm25_accepts_parallel_comma_separated_calls() -> None:
    text = (
        "<|tool_call_start|>"
        '[alpha(n=1, flag=true), beta(items=["a","b"], nested={k:null}), gamma()]'
        "<|tool_call_end|>"
    )
    output = normalize_tool_output(text, profile="lfm25-v1")

    assert [(call.index, call.name) for call in output.tool_calls] == [
        (0, "alpha"),
        (1, "beta"),
        (2, "gamma"),
    ]
    assert [call.arguments_json for call in output.tool_calls] == [
        {"n": 1, "flag": True},
        {"items": ["a", "b"], "nested": {"k": None}},
        {},
    ]


@pytest.mark.parametrize("profile", ["gemma4-v1", "lfm25-v1"])
def test_marker_profiles_do_not_guess_calls_from_prose(profile: str) -> None:
    output = normalize_tool_output("ordinary answer text", profile=profile)

    assert output.tool_calls == ()
    assert output.remaining_text == "ordinary answer text"


@pytest.mark.parametrize(
    ("profile", "text", "code"),
    [
        # Gemma 4 envelope and argument edge cases.
        ("gemma4-v1", "<|tool_call>read{path:\"a\"}", "invalid_tool_call_shape"),
        ("gemma4-v1", "<|tool_call>call:read", "invalid_tool_call_shape"),
        ("gemma4-v1", '<|tool_call>call:read{path:"a"', "malformed_arguments"),
        ("gemma4-v1", '<|tool_call>call:read{path:"a" path:"b"}', "malformed_arguments"),
        ("gemma4-v1", '<|tool_call>call:read{path:"a",path:"b"}', "duplicate_json_key"),
        (
            "gemma4-v1",
            '<|tool_call>call:bad name{path:"a"}',
            "invalid_tool_name",
        ),
        # LFM2.5 envelope and argument edge cases.
        ("lfm25-v1", "<|tool_call_start|><|tool_call_end|>", "invalid_tool_call_shape"),
        (
            "lfm25-v1",
            "<|tool_call_start|>[]<|tool_call_end|>",
            "empty_tool_section",
        ),
        (
            "lfm25-v1",
            '<|tool_call_start|>[read(path="a"<|tool_call_end|>',
            "invalid_tool_call_shape",
        ),
        (
            "lfm25-v1",
            '<|tool_call_start|>[bad name(path="a")]<|tool_call_end|>',
            "invalid_tool_name",
        ),
        (
            "lfm25-v1",
            '<|tool_call_start|>[read(path)]<|tool_call_end|>',
            "malformed_arguments",
        ),
        (
            "lfm25-v1",
            '<|tool_call_start|>[read(path="a" extra)]<|tool_call_end|>',
            "malformed_arguments",
        ),
        (
            "lfm25-v1",
            '<|tool_call_start|>[read(a=1, a=2)]<|tool_call_end|>',
            "duplicate_json_key",
        ),
        (
            "lfm25-v1",
            '<|tool_call_start|>[read(path=<bad>), write(x=1)]<|tool_call_end|>',
            "malformed_arguments",
        ),
    ],
)
def test_new_profiles_reject_malformed_or_ambiguous_calls(
    profile: str,
    text: str,
    code: str,
) -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(text, profile=profile)

    assert captured.value.code == code
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    ("limits", "text", "code"),
    [
        (
            ToolOutputLimits(max_calls=1),
            "<|tool_call>call:a{}<|tool_call>call:b{}",
            "call_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_json_depth=2),
            '<|tool_call>call:read{data:{deep:{x:1}}}',
            "json_depth_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_json_string_bytes=2),
            '<|tool_call>call:read{long:"abcdef"}',
            "json_string_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_calls=1),
            '<|tool_call_start|>[a(), b()]<|tool_call_end|>',
            "call_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_json_depth=2),
            '<|tool_call_start|>[read(data={deep:{x:1}})]<|tool_call_end|>',
            "json_depth_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_json_nodes=3),
            '<|tool_call_start|>[read(a={"b":{"c":1}})]<|tool_call_end|>',
            "json_node_limit_exceeded",
        ),
    ],
)
def test_new_profile_limits_fail_closed(
    limits: ToolOutputLimits,
    text: str,
    code: str,
) -> None:
    profile = NormalizationProfile.GEMMA4_V1 if "tool_call>" in text else NormalizationProfile.LFM25_V1
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(text, profile=profile, limits=limits)

    assert captured.value.code == code


def test_lfm25_rejects_ambiguous_multiple_sections() -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            "<|tool_call_start|><|tool_call_start|><|tool_call_end|>",
            profile=NormalizationProfile.LFM25_V1,
        )

    assert captured.value.code == "ambiguous_tool_section"

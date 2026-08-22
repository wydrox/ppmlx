"""Tests for the Qwen3.5 (qwen35-toolcall-v1) XML tool-call profile."""
from __future__ import annotations

import json

import pytest

from ppmlx.local_runtime.normalization import (
    NormalizationProfile,
    ToolNormalizationError,
    normalize_tool_output,
)


def test_qwen35_parses_single_call_with_leading_text() -> None:
    text = (
        "Let me look that up.\n"
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\n"
        "Paris\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    output = normalize_tool_output(text, profile=NormalizationProfile.QWEN35_TOOLCALL_V1)

    assert output.profile is NormalizationProfile.QWEN35_TOOLCALL_V1
    assert output.remaining_text == "Let me look that up.\n"
    assert len(output.tool_calls) == 1
    call = output.tool_calls[0]
    assert call.index == 0
    assert call.name == "get_weather"
    assert call.arguments_json == {"city": "Paris"}
    assert json.loads(call.arguments_raw) == call.arguments_json
    assert call.call_id is None
    assert call.repair is None


def test_qwen35_accepts_multiple_function_blocks_in_one_section() -> None:
    text = (
        "<tool_call>\n"
        "<function=get_weather>\n"
        "<parameter=city>\n"
        "Paris\n"
        "</parameter>\n"
        "</function>\n"
        "<function=get_time>\n"
        "<parameter=timezone>\n"
        "Europe/Warsaw\n"
        "</parameter>\n"
        "<parameter=format>\n"
        "24h\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    output = normalize_tool_output(text, profile="qwen35-toolcall-v1")

    assert [(call.index, call.name) for call in output.tool_calls] == [
        (0, "get_weather"),
        (1, "get_time"),
    ]
    assert [call.arguments_json for call in output.tool_calls] == [
        {"city": "Paris"},
        {"timezone": "Europe/Warsaw", "format": "24h"},
    ]


def test_qwen35_keeps_multiline_parameter_value_but_strips_one_newline() -> None:
    text = (
        "<tool_call>"
        "<function=write_file>"
        "<parameter=path>"
        "src/app.py\n"
        "</parameter>"
        "<parameter=content>"
        "line one\nline two\n"
        "</parameter>"
        "</function>"
        "</tool_call>"
    )
    output = normalize_tool_output(text, profile="qwen35-toolcall-v1")

    assert output.tool_calls[0].arguments_json == {
        "path": "src/app.py",
        "content": "line one\nline two",
    }


def test_qwen35_rejects_missing_function_close_tag() -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            "<tool_call><function=get_weather><parameter=city>Paris</parameter></tool_call>",
            profile="qwen35-toolcall-v1",
        )

    assert captured.value.code == "unterminated_tool_call"


def test_qwen35_rejects_malformed_block() -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            "<tool_call>prose without a function block</tool_call>",
            profile="qwen35-toolcall-v1",
        )

    assert captured.value.code == "malformed_tool_section"


def test_qwen35_rejects_unknown_tool_name() -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            "<tool_call><function=not a name!></function></tool_call>",
            profile="qwen35-toolcall-v1",
        )

    assert captured.value.code == "invalid_tool_name"


def test_qwen35_rejects_empty_section() -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output("<tool_call>\n</tool_call>", profile="qwen35-toolcall-v1")

    assert captured.value.code == "empty_tool_section"


def test_qwen35_does_not_guess_calls_from_prose() -> None:
    output = normalize_tool_output("ordinary answer text", profile="qwen35-toolcall-v1")

    assert output.tool_calls == ()
    assert output.remaining_text == "ordinary answer text"

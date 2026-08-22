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


def _typed_call(arguments: dict[str, object]):
    def _render(value: object) -> str:
        # Non-strings are rendered as JSON literals; strings are emitted raw
        # so we can also exercise values that must stay plain text.
        return value if isinstance(value, str) else json.dumps(value)

    lines = "".join(
        f"<parameter={key}>\n{_render(value)}\n</parameter>\n"
        for key, value in arguments.items()
    )
    text = f"<tool_call>\n<function=do_thing>\n{lines}</function>\n</tool_call>"
    output = normalize_tool_output(text, profile=NormalizationProfile.QWEN35_TOOLCALL_V1)
    assert len(output.tool_calls) == 1
    return output.tool_calls[0]


def test_qwen35_type_inference_kept_strings() -> None:
    call = _typed_call({"city": "Paris", "unit": "celsius", "note": "[a, b]"})
    assert call.arguments_json == {"city": "Paris", "unit": "celsius", "note": "[a, b]"}
    assert all(isinstance(v, str) for v in call.arguments_json.values())


def test_qwen35_type_inference_scalars() -> None:
    call = _typed_call(
        {"line": 7, "rate": 0.025, "flag": True, "missing": None}
    )
    args = call.arguments_json
    assert args == {"line": 7, "rate": 0.025, "flag": True, "missing": None}
    assert type(args["line"]) is int
    assert type(args["rate"]) is float
    assert type(args["flag"]) is bool
    assert args["missing"] is None


def test_qwen35_type_inference_containers() -> None:
    call = _typed_call(
        {
            "tags": ["bug", "phase-4"],
            "config": {"extensions": ["py"]},
        }
    )
    assert call.arguments_json == {
        "tags": ["bug", "phase-4"],
        "config": {"extensions": ["py"]},
    }


def test_qwen35_type_inference_invalid_json_object_text_stays_string() -> None:
    body = (
        "<tool_call>\n<function=f>\n<parameter=snippet>\n"
        "{not json\nsecond line}\n</parameter>\n</function>\n</tool_call>"
    )
    output = normalize_tool_output(body, profile=NormalizationProfile.QWEN35_TOOLCALL_V1)
    assert output.tool_calls[0].arguments_json == {
        "snippet": "{not json\nsecond line}"
    }


def test_qwen35_none_literal_is_not_null() -> None:
    call = _typed_call({"py_value": "None"})
    assert call.arguments_json == {"py_value": "None"}


def _envelope(name: str, parameter: str, value: str) -> str:
    return (
        "<tool_call>\n"
        f"<function={name}>\n"
        f"<parameter={parameter}>\n"
        f"{value}\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def test_qwen35_parses_two_consecutive_envelopes_in_order() -> None:
    text = _envelope("read_document", "path", "README.md") + _envelope(
        "read_document", "path", "SECURITY.md"
    )
    output = normalize_tool_output(text, profile=NormalizationProfile.QWEN35_TOOLCALL_V1)

    assert output.remaining_text == ""
    assert [(call.index, call.name) for call in output.tool_calls] == [
        (0, "read_document"),
        (1, "read_document"),
    ]
    assert output.tool_calls[0].arguments_json == {"path": "README.md"}
    assert output.tool_calls[1].arguments_json == {"path": "SECURITY.md"}


def test_qwen35_parses_three_consecutive_envelopes_in_order() -> None:
    text = (
        _envelope("get_weather", "city", "Paris")
        + _envelope("get_weather", "city", "London")
        + _envelope("get_time", "timezone", "Europe/Warsaw")
    )
    output = normalize_tool_output(text, profile=NormalizationProfile.QWEN35_TOOLCALL_V1)

    assert [(call.index, call.name) for call in output.tool_calls] == [
        (0, "get_weather"),
        (1, "get_weather"),
        (2, "get_time"),
    ]
    assert output.tool_calls[2].arguments_json == {"timezone": "Europe/Warsaw"}


def test_qwen35_rejects_text_between_envelopes() -> None:
    text = (
        _envelope("get_weather", "city", "Paris")
        + "\nSome prose in between.\n"
        + _envelope("get_weather", "city", "London")
    )
    with pytest.raises(ToolNormalizationError) as excinfo:
        normalize_tool_output(text, profile=NormalizationProfile.QWEN35_TOOLCALL_V1)
    assert excinfo.value.code == "text_after_tool_section"


def test_qwen35_single_envelope_still_works_after_multi_envelope_change() -> None:
    text = "Sure.\n" + _envelope("get_weather", "city", "Paris")
    output = normalize_tool_output(text, profile=NormalizationProfile.QWEN35_TOOLCALL_V1)

    assert output.remaining_text == "Sure.\n"
    assert [(call.index, call.name) for call in output.tool_calls] == [
        (0, "get_weather")
    ]

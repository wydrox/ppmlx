"""Tests for strict local model tool-output profiles."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from ppmlx.local_runtime.normalization import (
    NormalizationProfile,
    ToolNormalizationError,
    ToolOutputLimits,
    normalize_tool_output,
)


ROOT = Path(__file__).resolve().parents[1]


def test_qwen_json_keeps_exact_argument_lexeme_and_leading_text() -> None:
    arguments = '{ "path" : "a/b", "line" : 7 }'
    output = normalize_tool_output(
        'I will inspect the file.\n<tool_call>\n{"name":"read","arguments":'
        + arguments
        + "}\n</tool_call>",
        profile=NormalizationProfile.QWEN_JSON_V1,
    )

    assert output.profile is NormalizationProfile.QWEN_JSON_V1
    assert output.remaining_text == "I will inspect the file.\n"
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].name == "read"
    assert output.tool_calls[0].arguments_raw == arguments
    assert output.tool_calls[0].arguments_json == {"path": "a/b", "line": 7}
    assert output.tool_calls[0].call_id is None


def test_qwen_json_accepts_ordered_multiple_calls() -> None:
    output = normalize_tool_output(
        '<tool_call>{"name":"first","arguments":{"n":1}}</tool_call>\n'
        '<tool_call>{"name":"second","arguments":{"n":2}}</tool_call>',
        profile="qwen-json-v1",
    )

    assert [(call.index, call.name) for call in output.tool_calls] == [(0, "first"), (1, "second")]


def test_kimi_keeps_exact_arguments_and_native_identifier() -> None:
    arguments = '{"city": "Warsaw", "units":"celsius"}'
    text = (
        "Checking now."
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.get_weather:0"
        "<|tool_call_argument_begin|>"
        + arguments
        + "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    output = normalize_tool_output(text, profile=NormalizationProfile.KIMI_K2_V1)

    assert output.remaining_text == "Checking now."
    assert output.tool_calls[0].call_id == "functions.get_weather:0"
    assert output.tool_calls[0].name == "get_weather"
    assert output.tool_calls[0].arguments_raw == arguments
    assert output.tool_calls[0].arguments_json == {"city": "Warsaw", "units": "celsius"}


def test_kimi_accepts_parallel_calls_with_unique_identifiers() -> None:
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.alpha:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
        "<|tool_call_begin|>functions.beta:1<|tool_call_argument_begin|>{}<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    output = normalize_tool_output(text, profile="kimi-k2-v1")

    assert [call.call_id for call in output.tool_calls] == ["functions.alpha:0", "functions.beta:1"]


def test_deepseek_keeps_exact_fenced_json_arguments() -> None:
    arguments = '{"query": "safe value", "limit":  3}'
    text = (
        "I will use the search tool."
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>function<｜tool▁sep｜>search\n```json\n"
        + arguments
        + "\n```<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )
    output = normalize_tool_output(text, profile=NormalizationProfile.DEEPSEEK_V3_V1)

    assert output.remaining_text == "I will use the search tool."
    assert output.tool_calls[0].name == "search"
    assert output.tool_calls[0].arguments_raw == arguments
    assert output.tool_calls[0].arguments_json == {"query": "safe value", "limit": 3}


def test_grok_openai_chat_keeps_decoded_function_arguments() -> None:
    arguments = '{ "location": "Palo Alto", "unit": "fahrenheit" }'
    text = json.dumps(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_weather_01",
                    "type": "function",
                    "function": {"name": "get_temperature", "arguments": arguments},
                }
            ],
        }
    )
    output = normalize_tool_output(text, profile=NormalizationProfile.GROK_OPENAI_CHAT_V1)

    assert output.remaining_text == ""
    assert output.tool_calls[0].call_id == "call_weather_01"
    assert output.tool_calls[0].arguments_raw == arguments
    assert output.tool_calls[0].arguments_json == {"location": "Palo Alto", "unit": "fahrenheit"}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Final answer", "Final answer"),
        (
            '{"role":"assistant","content":"Final answer","tool_calls":[]}',
            "Final answer",
        ),
    ],
)
def test_grok_openai_chat_accepts_final_text(text: str, expected: str) -> None:
    output = normalize_tool_output(text, profile=NormalizationProfile.GROK_OPENAI_CHAT_V1)

    assert output.tool_calls == ()
    assert output.remaining_text == expected


@pytest.mark.parametrize(
    ("profile", "text"),
    [
        ("qwen-json-v1", "ordinary text"),
        ("kimi-k2-v1", "ordinary text"),
        ("deepseek-v3-v1", "ordinary text"),
    ],
)
def test_marker_profiles_do_not_guess_calls_from_prose(profile: str, text: str) -> None:
    output = normalize_tool_output(text, profile=profile)

    assert output.tool_calls == ()
    assert output.remaining_text == text


@pytest.mark.parametrize(
    ("profile", "text", "code"),
    [
        (
            "qwen-json-v1",
            '<tool_call>{"name":"read","arguments":{"x":1,"x":2}}</tool_call>',
            "duplicate_json_key",
        ),
        (
            "qwen-json-v1",
            '<tool_call>{"name":"read","name":"write","arguments":{}}</tool_call>',
            "duplicate_json_key",
        ),
        (
            "qwen-json-v1",
            '<tool_call>{"name":"read","arguments":[]}</tool_call>',
            "arguments_not_object",
        ),
        (
            "qwen-json-v1",
            '<tool_call>{"name":"read","arguments":{}}</tool_call> trailing text',
            "malformed_tool_section",
        ),
        (
            "qwen-json-v1",
            '<tool_call>{"name":"read","arguments":{}}',
            "unterminated_tool_call",
        ),
        (
            "kimi-k2-v1",
            "<|tool_calls_section_begin|><|tool_calls_section_end|>",
            "empty_tool_section",
        ),
        (
            "kimi-k2-v1",
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.read:0<|tool_call_argument_begin|>{}<|tool_call_end|>"
            "<|tool_calls_section_end|> trailing",
            "text_after_tool_section",
        ),
        (
            "deepseek-v3-v1",
            "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>read\n{}"
            "<｜tool▁call▁end｜><｜tool▁calls▁end｜>",
            "invalid_tool_call_shape",
        ),
        (
            "deepseek-v3-v1",
            "<｜tool▁calls▁begin｜><｜tool▁calls▁begin｜><｜tool▁calls▁end｜>",
            "ambiguous_tool_section",
        ),
        (
            "grok-openai-chat-v1",
            '{"role":"assistant","content":null,"tool_calls":['
            '{"id":"same","type":"function","function":{"name":"a","arguments":"{}"}},'
            '{"id":"same","type":"function","function":{"name":"b","arguments":"{}"}}]}',
            "duplicate_call_id",
        ),
        (
            "grok-openai-chat-v1",
            '{"role":"assistant","content":"prose","tool_calls":['
            '{"id":"call_1","type":"function","function":{"name":"a","arguments":"{}"}}]}',
            "ambiguous_tool_output",
        ),
        (
            "grok-openai-chat-v1",
            '{"role":"assistant","content":null,"tool_calls":['
            '{"id":"call_1","type":"function","function":{"name":"a","arguments":"{bad}"}}]}',
            "malformed_arguments",
        ),
    ],
)
def test_profiles_reject_malformed_duplicate_or_ambiguous_calls(
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
        (ToolOutputLimits(max_output_bytes=2), "ordinary text", "output_limit_exceeded"),
        (
            ToolOutputLimits(max_arguments_bytes=2),
            '<tool_call>{"name":"read","arguments":{"x":1}}</tool_call>',
            "arguments_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_calls=1),
            '<tool_call>{"name":"a","arguments":{}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{}}</tool_call>',
            "call_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_json_depth=2),
            '<tool_call>{"name":"read","arguments":{"x":{"y":1}}}</tool_call>',
            "json_depth_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_json_nodes=2),
            '<tool_call>{"name":"read","arguments":{"x":1}}</tool_call>',
            "json_node_limit_exceeded",
        ),
        (
            ToolOutputLimits(max_json_string_bytes=2),
            '<tool_call>{"name":"read","arguments":{"long":"x"}}</tool_call>',
            "json_string_limit_exceeded",
        ),
    ],
)
def test_limits_fail_closed(limits: ToolOutputLimits, text: str, code: str) -> None:
    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(text, profile="qwen-json-v1", limits=limits)

    assert captured.value.code == code


def test_error_does_not_keep_model_output_or_unknown_profile() -> None:
    secret = "credential-test-THIS_IS_SECRET_123456"

    with pytest.raises(ToolNormalizationError) as captured:
        normalize_tool_output(
            f'<tool_call>{{"name":"read","arguments":{{"secret":"{secret}"}},',
            profile=f"unknown-{secret}",
        )

    assert secret not in str(captured.value)
    assert secret not in captured.value.profile
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_normalization_module_has_only_pure_standard_library_imports() -> None:
    source = (ROOT / "ppmlx" / "local_runtime" / "normalization.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports.update(
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module != "__future__"
        and node.level == 0
    )

    assert imports <= {"dataclasses", "enum", "json", "re", "typing"}

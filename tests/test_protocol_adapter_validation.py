"""Security and limit tests for the shared protocol-adapter boundary."""
from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType

import pytest

import ppmlx.protocols as protocols
from ppmlx.protocols import (
    AdapterLimits,
    DecodeContext,
    ProtocolAdapter,
    ProtocolAdapterError,
    anthropic_messages_adapter,
    openai_chat_adapter,
    openai_responses_adapter,
)
from ppmlx.protocols.base import safe_adapter_boundary
from ppmlx.protocols.json import ensure_safe_evidence, parse_json_object
from ppmlx.protocols.sse import SSEFrame, encode_sse, encode_sse_frame, parse_sse


PROTOCOL = "test-protocol"
ROOT = Path(__file__).resolve().parents[1]


class _HostileMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("TOP-SECRET-MAPPING-CONTEXT")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("TOP-SECRET-MAPPING-CONTEXT")

    def __len__(self) -> int:
        return 1


@pytest.mark.parametrize("value", [
    '{"value":1,"value":2}',
    '{"value":NaN}',
    b"\xff",
    "[]",
    {"value": (1, 2)},
    {1: "value"},
])
def test_native_json_rejects_ambiguous_or_non_json_input(value: object) -> None:
    with pytest.raises(ProtocolAdapterError, match="invalid_json"):
        parse_json_object(value, protocol=PROTOCOL, limits=AdapterLimits())  # type: ignore[arg-type]


def test_native_json_returns_a_detached_plain_object() -> None:
    source = MappingProxyType({"value": [1, {"ok": True}]})
    parsed = parse_json_object(source, protocol=PROTOCOL, limits=AdapterLimits())
    assert parsed == {"value": [1, {"ok": True}]}
    assert type(parsed) is dict
    assert type(parsed["value"]) is list


@pytest.mark.parametrize("limits,value", [
    (AdapterLimits(max_request_bytes=2), '{"x":1}'),
    (AdapterLimits(max_string_bytes=2), {"x": "long"}),
    (AdapterLimits(max_json_depth=1), {"x": {"nested": True}}),
    (AdapterLimits(max_json_nodes=2), {"x": [1, 2]}),
])
def test_native_json_enforces_input_limits(limits: AdapterLimits, value: object) -> None:
    with pytest.raises(ProtocolAdapterError, match="invalid_json"):
        parse_json_object(value, protocol=PROTOCOL, limits=limits)  # type: ignore[arg-type]


def test_mapping_input_uses_the_aggregate_request_limit() -> None:
    with pytest.raises(ProtocolAdapterError, match="invalid_json"):
        parse_json_object(
            {"value": "long-value"},
            protocol=PROTOCOL,
            limits=AdapterLimits(max_request_bytes=8),
        )


@pytest.mark.parametrize("value", [
    {"authorization": "Bearer safe-looking"},
    {"access_token": "plain-secret-value"},
    {"access-token": "plain-secret-value"},
    {"client-secret": "plain-secret-value"},
    {"proxy_authorization": "plain-secret-value"},
    {"api_token": "plain-secret-value"},
    {"private_token": "plain-secret-value"},
    {"personal_access_token": "plain-secret-value"},
    {"secret_access_key": "plain-secret-value"},
    {"nested": {"api_key": "value"}},
    {"text": "Bearer secret-token-value"},
    {"text": "sk-test-THIS_IS_SECRET"},
])
def test_native_evidence_rejects_credential_data(value: object) -> None:
    with pytest.raises(ProtocolAdapterError, match="credential_in_evidence") as captured:
        ensure_safe_evidence(value, protocol=PROTOCOL)
    assert "secret" not in str(captured.value).lower()


def test_token_usage_fields_are_not_mistaken_for_credentials() -> None:
    ensure_safe_evidence(
        {"max_tokens": 20, "usage": {"input_tokens": 4, "output_tokens": 2}},
        protocol=PROTOCOL,
    )


def test_native_evidence_rejects_cycles() -> None:
    value: dict[str, object] = {}
    value["cycle"] = value

    with pytest.raises(ProtocolAdapterError, match="invalid_evidence"):
        ensure_safe_evidence(value, protocol=PROTOCOL)


def test_protocol_errors_do_not_retain_invalid_native_json() -> None:
    secret = "sk-test-THIS_IS_SECRET_123456"
    with pytest.raises(ProtocolAdapterError) as captured:
        parse_json_object(
            '{"value":"' + secret + '","value":2}',
            protocol=PROTOCOL,
            limits=AdapterLimits(),
        )
    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_protocol_error_text_does_not_expose_an_untrusted_field_name() -> None:
    secret = "sk-test-THIS_IS_SECRET_123456"
    error = ProtocolAdapterError(
        protocol=PROTOCOL,
        code="unsupported_request_field",
        field=f"/{secret}",
    )

    assert error.field is None
    assert secret not in str(error)


def test_safe_adapter_boundary_removes_a_private_validation_context() -> None:
    secret = "sk-test-THIS_IS_SECRET_123456"

    @safe_adapter_boundary
    def fail() -> None:
        try:
            raise ValueError(secret)
        except ValueError:
            raise ProtocolAdapterError(protocol=PROTOCOL, code="invalid_request") from None

    with pytest.raises(ProtocolAdapterError) as captured:
        fail()

    assert secret not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    ("adapter", "native"),
    [
        (
            anthropic_messages_adapter,
            {"model": "model", "messages": [], "sk-test-THIS_IS_SECRET_123456": True},
        ),
        (
            openai_chat_adapter,
            {
                "model": "model",
                "messages": [],
                "sk-test-THIS_IS_SECRET_123456": True,
            },
        ),
        (
            openai_responses_adapter,
            {"model": "model", "input": [], "sk-test-THIS_IS_SECRET_123456": True},
        ),
    ],
)
def test_public_adapters_do_not_retain_an_untrusted_field_name(
    adapter: ProtocolAdapter,
    native: dict[str, object],
) -> None:
    secret = "sk-test-THIS_IS_SECRET_123456"

    with pytest.raises(ProtocolAdapterError) as captured:
        adapter.decode_request(
            native,
            context=DecodeContext(request_id="req_secret_field", kind="initial"),
        )

    assert secret not in str(captured.value)
    assert secret not in (captured.value.field or "")
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    ("adapter", "native"),
    [
        (
            anthropic_messages_adapter,
            {
                "model": "model",
                "messages": [],
                "metadata": {"authorization": "Bearer private-token"},
            },
        ),
        (
            openai_chat_adapter,
            {
                "model": "model",
                "messages": [],
                "metadata": {"authorization": "Bearer private-token"},
            },
        ),
        (
            openai_responses_adapter,
            {
                "model": "model",
                "input": [],
                "client_metadata": {"authorization": "Bearer private-token"},
            },
        ),
    ],
)
def test_public_adapters_reject_credentials_when_native_evidence_is_off(
    adapter: ProtocolAdapter,
    native: dict[str, object],
) -> None:
    with pytest.raises(ProtocolAdapterError) as captured:
        adapter.decode_request(
            native,
            context=DecodeContext(request_id="req_secret_value", kind="initial"),
        )

    assert captured.value.code == "credential_in_evidence"
    assert "private-token" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.parametrize(
    "adapter",
    [anthropic_messages_adapter, openai_chat_adapter, openai_responses_adapter],
)
def test_public_adapters_sanitize_hostile_mapping_exceptions(
    adapter: ProtocolAdapter,
) -> None:
    with pytest.raises(ProtocolAdapterError) as captured:
        adapter.decode_request(
            _HostileMapping(),
            context=DecodeContext(request_id="req_hostile_mapping", kind="initial"),
        )

    assert "TOP-SECRET" not in str(captured.value)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


def test_capability_vocabulary_is_canonical_and_exact() -> None:
    common_request = {
        "generation",
        "instructions",
        "metadata",
        "stream",
        "text",
        "tool_calls",
        "tool_choice",
        "tool_results",
        "tools",
    }
    common_response = {"stream", "text", "tool_calls", "usage"}

    assert openai_chat_adapter.capabilities.request_features == common_request
    assert openai_responses_adapter.capabilities.request_features == common_request
    assert anthropic_messages_adapter.capabilities.request_features == common_request | {
        "document",
        "image",
    }
    for adapter in (
        anthropic_messages_adapter,
        openai_chat_adapter,
        openai_responses_adapter,
    ):
        assert adapter.capabilities.response_features == common_response


def test_public_protocol_package_has_a_small_stable_surface() -> None:
    assert set(protocols.__all__) == {
        "AdapterLimits",
        "AnthropicMessagesAdapter",
        "CallReference",
        "DecodeContext",
        "DecodedRequest",
        "EncodeContext",
        "NormalizationPolicy",
        "OpenAIChatAdapter",
        "OpenAIResponsesAdapter",
        "ProtocolAdapter",
        "ProtocolAdapterError",
        "ProtocolCapabilities",
        "anthropic_messages_adapter",
        "openai_chat_adapter",
        "openai_responses_adapter",
    }


def test_sse_helpers_preserve_event_names_json_and_terminal_sentinel() -> None:
    frames = (
        SSEFrame(event="response.created", data={"type": "response.created", "sequence": 0}),
        SSEFrame(event=None, data="[DONE]"),
    )
    encoded = encode_sse(frames)
    assert parse_sse(encoded, protocol=PROTOCOL) == frames


@pytest.mark.parametrize("value", [
    b"\xff",
    "event: x\n\n",
    "invalid: x\ndata: {}\n\n",
    'data: {"x":1,"x":2}\n\n',
    "data: NaN\n\n",
])
def test_sse_parser_rejects_invalid_or_ambiguous_frames(value: str | bytes) -> None:
    with pytest.raises(ProtocolAdapterError, match="invalid_sse"):
        parse_sse(value, protocol=PROTOCOL)


def test_sse_parser_enforces_frame_size_limit() -> None:
    with pytest.raises(ProtocolAdapterError, match="sse_frame_too_large"):
        parse_sse(
            'data: {"value":"long"}\n\n',
            protocol=PROTOCOL,
            limits=AdapterLimits(max_sse_frame_bytes=8),
        )


def test_sse_parser_enforces_total_size_and_frame_count_limits() -> None:
    stream = 'data: {"value":1}\n\ndata: {"value":2}\n\n'

    with pytest.raises(ProtocolAdapterError, match="sse_stream_too_large"):
        parse_sse(
            stream,
            protocol=PROTOCOL,
            limits=AdapterLimits(max_sse_stream_bytes=8),
        )
    with pytest.raises(ProtocolAdapterError, match="too_many_sse_frames"):
        parse_sse(
            stream,
            protocol=PROTOCOL,
            limits=AdapterLimits(max_events=1),
        )


def test_sse_encoder_stops_before_it_serializes_the_rest_of_an_oversize_stream() -> None:
    def frames() -> Iterator[SSEFrame]:
        yield SSEFrame(event=None, data={"value": "first"})
        raise AssertionError("The encoder read past the total stream limit")

    with pytest.raises(ProtocolAdapterError, match="sse_stream_too_large"):
        encode_sse(
            frames(),
            protocol=PROTOCOL,
            limits=AdapterLimits(max_sse_stream_bytes=1),
        )


@pytest.mark.parametrize(
    ("data", "event"),
    [
        ("safe\n\ndata: injected", None),
        ({"safe": True}, "safe\n\nevent: injected"),
    ],
)
def test_sse_encoder_rejects_frame_injection(data: object, event: str | None) -> None:
    with pytest.raises(ValueError):
        encode_sse_frame(data, event=event)


def test_protocol_modules_do_not_import_stateful_or_transport_code() -> None:
    forbidden_roots = {
        "httpx",
        "keyring",
        "os",
        "pathlib",
        "requests",
        "socket",
        "sqlite3",
        "subprocess",
        "urllib",
    }
    forbidden_ppmlx = {
        "ppmlx.config",
        "ppmlx.db",
        "ppmlx.engine",
        "ppmlx.memory_engine",
        "ppmlx.memory_store",
        "ppmlx.server",
    }
    for path in sorted((ROOT / "ppmlx" / "protocols").glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module}
            else:
                continue
            assert not {name.split(".", 1)[0] for name in names} & forbidden_roots, path
            assert not names & forbidden_ppmlx, path

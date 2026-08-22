"""Audit-fix tests for the remote OpenAI provider (H1, M3, L9)."""
from __future__ import annotations

import json

import httpx
import pytest

from ppmlx.protocols import DecodeContext, openai_chat_adapter
from ppmlx.providers import (
    OpenAIProvider,
    ProviderError,
    ProviderInvocation,
)

API_KEY = "sk-tes...3456"


def _json_handler(payload_or_bytes, *, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        if isinstance(payload_or_bytes, bytes):
            return httpx.Response(
                status,
                headers={"Content-Type": "application/json"},
                content=payload_or_bytes,
                request=request,
            )
        return httpx.Response(status, json=payload_or_bytes, request=request)

    return handler


def _invocation(**kwargs: object) -> ProviderInvocation:
    native = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "You there?"}],
    }
    envelope = openai_chat_adapter.decode_request(
        native,
        context=DecodeContext(request_id="req_audit", kind="initial"),
    ).request
    return ProviderInvocation(request=envelope, model_id="gpt-4o", **kwargs)


def _provider(handler, *, max_response_bytes: int = 10 * 1024 * 1024):
    return OpenAIProvider(
        env_key="PPMLX_AUDIT_TEST_KEY",
        max_response_bytes=max_response_bytes,
        transport=httpx.MockTransport(handler),
    )


# ----------------------------------------------------------------------
# H1: size cap enforced during buffered read
# ----------------------------------------------------------------------


def test_h1_oversized_buffered_body_rejected_during_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PPMLX_AUDIT_TEST_KEY", API_KEY)
    # A valid completion document that is larger than the configured cap:
    # under the old implementation this was fully read into memory first
    # and only rejected afterwards.
    document = {
        "id": "chatcmpl_big",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "x" * 4096},
             "finish_reason": "stop"}
        ],
    }
    body = json.dumps(document).encode("utf-8")
    provider = _provider(_json_handler(body), max_response_bytes=1024)

    with pytest.raises(ProviderError) as excinfo:
        provider.invoke(_invocation())

    assert excinfo.value.code == "response_too_large"
    assert len(body) > 1024


def test_h1_undersized_buffered_body_still_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PPMLX_AUDIT_TEST_KEY", API_KEY)
    document = {
        "id": "chatcmpl_ok",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hi"},
             "finish_reason": "stop"}
        ],
    }
    provider = _provider(_json_handler(document), max_response_bytes=65536)
    result = provider.invoke(_invocation())
    assert result.cancelled is False
    assert any(event.type == "content.completed" for event in result.events)


# ----------------------------------------------------------------------
# M3: sanitized diagnostic detail, chained causes, clean messages
# ----------------------------------------------------------------------


def test_m3_invalid_json_detail_is_sanitized_and_chained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PPMLX_AUDIT_TEST_KEY", API_KEY)
    secret_marker = b"sk-super-secret-value"
    body = b'{"choices": [' + secret_marker + b' broken json'
    provider = _provider(_json_handler(body))

    with pytest.raises(ProviderError) as excinfo:
        provider.invoke(_invocation())

    err = excinfo.value
    assert err.code == "invalid_response"
    # Detail present but sanitized: exception class + byte count only.
    assert err.detail is not None
    assert "JSONDecodeError" in err.detail or "UnicodeDecodeError" in err.detail
    assert f"bytes={len(body)}" in err.detail
    # Never leaks body content or secrets.
    assert secret_marker.decode() not in (err.detail or "")
    assert secret_marker.decode() not in str(err)
    # Message stays clean.
    assert "openai provider error invalid_response" == str(err)
    # Cause is chained instead of discarded.
    assert err.__cause__ is not None


def test_m3_transport_error_carries_exception_class_detail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PPMLX_AUDIT_TEST_KEY", API_KEY)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    provider = _provider(handler)
    with pytest.raises(ProviderError) as excinfo:
        provider.invoke(_invocation())
    err = excinfo.value
    assert err.code == "network_error"
    assert err.detail == "exception=ConnectError"
    assert isinstance(err.__cause__, httpx.TransportError)


# ----------------------------------------------------------------------
# L9: buffered path rejects != 1 choices (parity with streaming)
# ----------------------------------------------------------------------


def _completion_with_choices(choices: list[dict[str, object]]) -> dict[str, object]:
    return {
        "id": "chatcmpl_multi",
        "object": "chat.completion",
        "choices": choices,
    }


def test_l9_buffered_rejects_multiple_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PPMLX_AUDIT_TEST_KEY", API_KEY)
    document = _completion_with_choices(
        [
            {"index": 0, "message": {"role": "assistant", "content": "a"},
             "finish_reason": "stop"},
            {"index": 1, "message": {"role": "assistant", "content": "b"},
             "finish_reason": "stop"},
        ]
    )
    provider = _provider(_json_handler(document))
    with pytest.raises(ProviderError) as excinfo:
        provider.invoke(_invocation())
    assert excinfo.value.code == "invalid_response"


def test_l9_streaming_rejects_nonzero_choice_index_parity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PPMLX_AUDIT_TEST_KEY", API_KEY)
    chunk = {
        "id": "chatcmpl_sse",
        "object": "chat.completion.chunk",
        "choices": [
            {"index": 1, "delta": {"content": "nope"}, "finish_reason": None}
        ],
    }
    frame = b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\ndata: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=frame,
            request=request,
        )

    provider = _provider(handler)
    with pytest.raises(ProviderError) as excinfo:
        list(provider.stream(_invocation()))
    assert excinfo.value.code == "invalid_response"

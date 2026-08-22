"""Tests for the Claude Code subscription passthrough provider."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
import pytest

from ppmlx.agent_ir import (
    ContentDeltaEvent,
    ContentStartedEvent,
    ResponseCompletedEvent,
)
from ppmlx.protocols import DecodeContext, anthropic_messages_adapter
from ppmlx.providers import (
    Provider,
    ProviderCancellationHandle,
    ProviderCancelledError,
    ProviderCredentialType,
    ProviderDataPath,
    ProviderError,
    ProviderHealthStatus,
    ProviderInvocation,
)
from ppmlx.providers.anthropic_subscription import (
    LockedSubscriptionPassthrough,
    SubscriptionCredentialsUnavailable,
    SubscriptionPassthroughProvider,
    read_claude_code_oauth_token,
)

TOKEN = "sk-ant-oat01-testtoken123"


def _write_credentials(
    tmp_path: Path,
    token: str = TOKEN,
    *,
    shape: str = "claudeAiOauth",
) -> Path:
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    if shape == "claudeAiOauth":
        document = {
            "claudeAiOauth": {
                "accessToken": token,
                "refreshToken": "refresh",
                "expiresAt": 9999999999999,
            }
        }
    elif shape == "topLevel":
        document = {"accessToken": token}
    else:
        document = {"oauth": {"accessToken": token}}
    (claude_dir / ".credentials.json").write_text(json.dumps(document))
    return claude_dir


def _provider(claude_dir: Path, transport: httpx.BaseTransport, **kwargs) -> SubscriptionPassthroughProvider:
    return SubscriptionPassthroughProvider(
        claude_dir=claude_dir,
        transport=transport,
        base_url="https://api.anthropic.com",
        **kwargs,
    )


def _invocation(**kwargs) -> ProviderInvocation:
    native = {
        "model": "claude-sonnet-4-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "You there?"}]}
        ],
    }
    envelope = anthropic_messages_adapter.decode_request(
        native,
        context=DecodeContext(request_id="req_sub_test", kind="initial"),
    ).request
    return ProviderInvocation(request=envelope, model_id="claude-sonnet-4-5", **kwargs)


# ----------------------------------------------------------------------
# Credential live-read
# ----------------------------------------------------------------------


def test_live_read_parses_known_credential_shapes(tmp_path: Path) -> None:
    for shape in ("claudeAiOauth", "topLevel", "nested"):
        assert read_claude_code_oauth_token(_write_credentials(tmp_path, shape=shape)) == TOKEN


def test_missing_credentials_file_is_typed_error_without_token(tmp_path: Path) -> None:
    with pytest.raises(SubscriptionCredentialsUnavailable):
        read_claude_code_oauth_token(tmp_path / ".claude")


def test_changed_format_is_typed_error(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path, token="not-an-oat-token")
    with pytest.raises(SubscriptionCredentialsUnavailable):
        read_claude_code_oauth_token(claude_dir)
    (claude_dir / ".credentials.json").write_text("this is not json {{{")
    with pytest.raises(SubscriptionCredentialsUnavailable):
        read_claude_code_oauth_token(claude_dir)


def test_token_rotation_is_picked_up_per_request(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json={"content": [], "stop_reason": "end_turn"}, request=request)

    provider = _provider(claude_dir, httpx.MockTransport(handler))
    provider.invoke(_invocation())
    _write_credentials(tmp_path, token="sk-ant-oat01-ROTATED")
    provider.invoke(_invocation())

    assert seen[0].endswith(TOKEN)
    assert seen[1].endswith("sk-ant-oat01-ROTATED")


def test_health_reports_unavailable_when_credentials_missing(tmp_path: Path) -> None:
    provider = _provider(tmp_path / "nope", httpx.MockTransport(lambda r: httpx.Response(200)))
    assert provider.health().status is ProviderHealthStatus.UNAVAILABLE
    assert provider.health().code == "credentials_unavailable"


# ----------------------------------------------------------------------
# Full tunnel: headers/body preservation and SSE passthrough
# ----------------------------------------------------------------------


def test_forward_preserves_headers_and_body_byte_for_byte(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = request.read()
        return httpx.Response(200, content=b'{"ok":true}', request=request)

    provider = _provider(claude_dir, httpx.MockTransport(handler))
    body = b'{"model":"claude-sonnet-4-5","messages":[],"stream":true,"x":1}'
    tunnel = provider.forward(
        "/v1/messages",
        headers={
            "user-agent": "claude-cli/2.1.0",
            "x-app": "cli",
            "anthropic-beta": "prompt-caching-2024-07-31",
            "x-stainless-lang": "js",
            "anthropic-version": "2023-06-01",
            "host": "ignored.example",
            "authorization": "Bearer stale",
        },
        body=body,
    )
    assert b"".join(tunnel.iter_bytes()) == b'{"ok":true}'
    assert captured["url"] == "https://api.anthropic.com/v1/messages"
    headers = captured["headers"]
    assert headers["user-agent"] == "claude-cli/2.1.0"
    assert headers["x-app"] == "cli"
    assert headers["anthropic-beta"] == "prompt-caching-2023-06-01" or headers[
        "anthropic-beta"
    ] == "prompt-caching-2024-07-31"
    assert headers["x-stainless-lang"] == "js"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["authorization"].endswith(TOKEN)
    assert "stale" not in headers["authorization"]
    # The client-derived host header points at the upstream origin, never at
    # the original caller's Host value.
    assert headers["host"] == "api.anthropic.com"
    assert captured["body"] == body


def test_sse_passthrough_preserves_chunk_order(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    frames = b"".join(
        (
            b'data: {"type":"message_start"}\n\n',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n\n',
            b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n\n',
            b'data: {"type":"message_stop"}\n\n',
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, content=frames, request=request
        )

    provider = _provider(claude_dir, httpx.MockTransport(handler))
    tunnel = provider.forward("/v1/messages", body=b"{}")
    chunks = list(tunnel.iter_bytes())
    joined = b"".join(chunks)
    assert joined == frames
    # Chunk-for-chunk order: each frame boundary appears in upstream order.
    assert joined.index(b'"text":"Hel"') < joined.index(b'"text":"lo"')
    assert joined.index(b'"text":"lo"') < joined.index(b'"message_stop"')


def test_forward_enforces_response_size_limit(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    provider = _provider(
        claude_dir,
        httpx.MockTransport(lambda r: httpx.Response(200, content=b"x" * 100, request=r)),
    )
    tunnel = provider.forward("/v1/messages", body=b"{}", max_response_bytes=10)
    with pytest.raises(ProviderError) as excinfo:
        list(tunnel.iter_bytes())
    assert excinfo.value.code == "response_too_large"


def test_forward_rejects_non_v1_paths(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    provider = _provider(claude_dir, httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(ProviderError):
        provider.forward("/admin/secret", body=b"")


def test_cancellation_mid_stream_raises_typed_error(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    gate = threading.Event()
    handle = ProviderCancellationHandle()

    def handler(request: httpx.Request) -> httpx.Response:
        def gen():
            yield b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"a"}}\n\n'
            gate.wait(timeout=5)
            yield b'data: {"type":"message_stop"}\n\n'
        return httpx.Response(200, content=gen(), request=request)

    provider = _provider(claude_dir, httpx.MockTransport(handler))
    tunnel = provider.forward("/v1/messages", body=b"{}", cancel_handle=handle)
    iterator = tunnel.iter_bytes()
    first = next(iterator)
    assert b"text_delta" in first
    handle.cancel()
    with pytest.raises(ProviderCancelledError):
        next(iterator)
    gate.set()


# ----------------------------------------------------------------------
# Provider protocol path (Agent IR invoke/stream via OAuth bearer)
# ----------------------------------------------------------------------


def test_provider_protocol_declares_oauth_remote_capabilities(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    provider = _provider(claude_dir, httpx.MockTransport(lambda r: httpx.Response(200)))
    assert isinstance(provider, Provider)
    caps = provider.capabilities("claude-sonnet-4-5")
    assert caps.data_path is ProviderDataPath.REMOTE
    assert caps.credential_types == (ProviderCredentialType.OAUTH_SESSION,)


def test_invoke_uses_bearer_token_and_returns_events(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    seen_auth: list[str] = []
    payload = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "Hello"}],
        "stop_reason": "end_turn",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization", ""))
        return httpx.Response(200, json=payload, request=request)

    provider = _provider(claude_dir, httpx.MockTransport(handler))
    result = provider.invoke(_invocation())
    kinds = [event.type for event in result.events]
    assert "response.completed" in kinds
    assert seen_auth == [f"Bearer {TOKEN}"]


def test_stream_emits_native_deltas_in_order(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)

    def frame(payload: dict) -> bytes:
        return b"data: " + json.dumps(payload).encode() + b"\n\n"

    body = b"".join(
        (
            frame({"type": "message_start", "message": {"usage": {"input_tokens": 1}}}),
            frame(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text"},
                }
            ),
            frame(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hi"},
                }
            ),
            frame({"type": "content_block_stop", "index": 0}),
            frame(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1},
                }
            ),
            frame({"type": "message_stop"}),
        )
    )
    provider = _provider(
        claude_dir,
        httpx.MockTransport(
            lambda r: httpx.Response(
                200, headers={"Content-Type": "text/event-stream"}, content=body, request=r
            )
        ),
    )
    events = list(provider.stream(_invocation()))
    deltas = [event.delta for event in events if isinstance(event, ContentDeltaEvent)]
    assert deltas == ["Hi"]
    assert isinstance(events[-1], ResponseCompletedEvent)


# ----------------------------------------------------------------------
# Memory capture hook
# ----------------------------------------------------------------------


def test_memory_capture_receives_redacted_content_and_provenance(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)
    secret = "sk-ant-api99-VERYSECRETVALUE"
    calls: list[dict] = []

    provider = _provider(
        claude_dir,
        httpx.MockTransport(lambda r: httpx.Response(200, content=b'{"ok":1}', request=r)),
    )
    provider.capture_hook = lambda **payload: calls.append(payload)
    provider.capture_to_memory(
        request_id="req_sub_capture",
        model_alias="claude-sonnet-4-5",
        request_text=f"user prompt with {secret} inside",
        response_text=f"answer with {secret} inside",
    )
    assert len(calls) == 1
    call = calls[0]
    assert call["metadata"]["source"] == "anthropic-subscription"
    assert call["model_repo"] == "anthropic-subscription"
    joined = json.dumps(call)
    assert secret not in joined
    assert "user prompt with" in call["request_text"]


def test_token_never_appears_in_errors_or_captured_output(tmp_path: Path) -> None:
    claude_dir = _write_credentials(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad token"}, request=request)

    provider = _provider(claude_dir, httpx.MockTransport(handler))
    with pytest.raises(ProviderError) as excinfo:
        provider.invoke(_invocation())
    blob = f"{excinfo.value}{excinfo.value.code}{getattr(excinfo.value, 'detail', '')}"
    assert TOKEN not in blob

    calls: list[dict] = []
    provider.capture_hook = lambda **payload: calls.append(payload)

    def big_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 64, request=request)

    provider._transport = httpx.MockTransport(big_handler)
    tunnel = provider.forward("/v1/messages", body=b"{}", max_response_bytes=8)
    with pytest.raises(ProviderError):
        list(tunnel.iter_bytes())
    for call in calls:
        assert TOKEN not in json.dumps(call)


# ----------------------------------------------------------------------
# Locked-by-default gate
# ----------------------------------------------------------------------


def test_locked_provider_names_config_flag() -> None:
    locked = LockedSubscriptionPassthrough()
    assert locked.list_models() == ()
    assert locked.health().status is ProviderHealthStatus.UNAVAILABLE
    assert locked.health().code == "subscription_passthrough_disabled"
    with pytest.raises(ProviderError) as excinfo:
        locked.capabilities("claude-sonnet-4-5")
    assert excinfo.value.code == "subscription_passthrough_disabled"
    assert "[dangerous] subscription_passthrough" in (excinfo.value.detail or "")
    with pytest.raises(ProviderError):
        locked.invoke(_invocation())


def test_config_gate_defaults_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    from ppmlx.config import Config

    config = Config()
    assert config.dangerous.subscription_passthrough is False


def test_config_gate_env_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from ppmlx.config import DangerousConfig

    monkeypatch.setenv("PPMLX_DANGEROUS_SUBSCRIPTION_PASSTHROUGH", "1")
    from ppmlx.config import _apply_env, Config

    config = Config()
    _apply_env(config)
    assert config.dangerous.subscription_passthrough is True

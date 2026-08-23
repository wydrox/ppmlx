"""Subscription passthrough provider: full tunnel to api.anthropic.com.

This provider transparently forwards requests to the Anthropic API using the
OAuth credentials of an installed, logged-in Claude Code installation. The
token is LIVE-READ from ``~/.claude/.credentials.json`` on every request so
Claude Code token rotation is picked up automatically; it is never copied
into ppmlx storage, keyring, logs, or error messages.

The feature is LOCKED by default behind ``[dangerous] subscription_passthrough``
in the ppmlx config. When disabled, the provider is not registered for
routing and any route targeting it fails with a typed error naming the flag.
"""
from __future__ import annotations

import json
import sys
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable

import httpx

from ppmlx.providers.anthropic import (
    DEFAULT_MAX_RESPONSE_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    AnthropicProvider,
)
from ppmlx.providers.base import (
    ProviderCapabilities,
    ProviderCancellationHandle,
    ProviderCancelledError,
    ProviderCredentialType,
    ProviderDataPath,
    ProviderError,
    ProviderHealth,
    ProviderHealthStatus,
)

SUBSCRIPTION_BASE_URL = "https://api.anthropic.com"
DEFAULT_CLAUDE_DIR = Path.home() / ".claude"
DEFAULT_CREDENTIALS_FILENAME = ".credentials.json"
OAUTH_TOKEN_PREFIX = "sk-ant-oat01"
PROVENANCE_SOURCE = "anthropic-subscription"

# Hop-by-hop / rewritten-by-the-HTTP-client headers that must NOT be forwarded
# verbatim. Everything else (user-agent, x-app, anthropic-beta, x-stainless-*,
# anthropic-version, ...) is preserved byte-for-byte.
_TUNNEL_DROP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
        "te",
        "trailer",
    }
)

_WARN_LOCK = threading.Lock()
_WARNED_THIS_SESSION = False

TOS_WARNING = (
    "subscription passthrough is ENABLED ([dangerous] subscription_passthrough): "
    "requests will be sent to api.anthropic.com under your personal Claude "
    "subscription credentials. This may violate the Anthropic Terms of Service "
    "or your plan's usage policy; you assume all responsibility for this use."
)


def warn_subscription_tos_once(stream: Any = None) -> None:
    """Print the ToS warning at most once per process/session."""
    global _WARNED_THIS_SESSION
    with _WARN_LOCK:
        if _WARNED_THIS_SESSION:
            return
        _WARNED_THIS_SESSION = True
    target = stream if stream is not None else sys.stderr
    try:
        print(f"WARNING: {TOS_WARNING}", file=target)
    except Exception:
        pass


class SubscriptionCredentialsUnavailable(ProviderError):
    """Typed error when Claude Code credentials are missing or unreadable.

    Never carries the token value or file contents in the message.
    """

    def __init__(self, *, reason: str) -> None:
        super().__init__(
            provider_id=PROVENANCE_SOURCE,
            code="credentials_unavailable",
            detail=reason,
        )
        self.reason = reason


def read_claude_code_oauth_token(
    claude_dir: Path | str = DEFAULT_CLAUDE_DIR,
) -> str:
    """Live-read the Claude Code OAuth access token from disk.

    Defensive parsing of the known credential-file formats. Raises
    :class:`SubscriptionCredentialsUnavailable` with a clear, token-free
    reason when the file is missing, unparsable, or has changed format.
    """
    directory = Path(claude_dir)
    path = directory / DEFAULT_CREDENTIALS_FILENAME
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SubscriptionCredentialsUnavailable(
            reason=(
                f"no Claude Code credentials file at {path}; log in with "
                "`claude` first or set [dangerous] subscription_passthrough "
                "expectations accordingly"
            )
        ) from None
    except OSError:
        raise SubscriptionCredentialsUnavailable(
            reason=f"Claude Code credentials file at {path} could not be read"
        ) from None
    try:
        document = json.loads(raw)
    except json.JSONDecodeError:
        raise SubscriptionCredentialsUnavailable(
            reason=(
                f"Claude Code credentials file at {path} is not valid JSON; "
                "the format may have changed"
            )
        ) from None
    token = _extract_oauth_token(document)
    if token is None:
        raise SubscriptionCredentialsUnavailable(
            reason=(
                f"Claude Code credentials file at {path} does not contain a "
                "recognized claudeAiOauth accessToken; the format may have "
                "changed"
            )
        )
    if not token.startswith(OAUTH_TOKEN_PREFIX):
        raise SubscriptionCredentialsUnavailable(
            reason=(
                "Claude Code access token has an unrecognized format; "
                "refusing to send it"
            )
        )
    return token


def _extract_oauth_token(document: Any) -> str | None:
    """Pull the OAuth access token out of known credential document shapes."""
    if not isinstance(document, Mapping):
        return None
    oauth = document.get("claudeAiOauth")
    if isinstance(oauth, Mapping):
        token = oauth.get("accessToken")
        if type(token) is str and token:
            return token
    token = document.get("accessToken")
    if type(token) is str and token:
        return token
    nested = document.get("oauth")
    if isinstance(nested, Mapping):
        token = nested.get("accessToken")
        if type(token) is str and token:
            return token
    return None


class TunnelResponse:
    """Raw upstream tunnel response streamed chunk-for-chunk."""

    __slots__ = ("status_code", "headers", "_chunks")

    def __init__(
        self,
        *,
        status_code: int,
        headers: Mapping[str, str],
        chunks: Iterator[bytes],
    ) -> None:
        self.status_code = status_code
        self.headers = dict(headers)
        self._chunks = chunks

    def iter_bytes(self) -> Iterator[bytes]:
        return self._chunks


class SubscriptionPassthroughProvider(AnthropicProvider):
    """Full-tunnel provider backed by Claude Code subscription credentials.

    Inherits the tested Agent IR encoding/SSE handling from
    :class:`AnthropicProvider`, replacing API-key auth with a per-request
    live-read OAuth bearer token. ``forward()`` additionally exposes the raw
    byte-for-byte tunnel used by the gateway passthrough route: original
    headers and body are preserved exactly and the upstream response is
    streamed back 1:1 (SSE included) with a size limit and cancellation.
    """

    def __init__(
        self,
        *,
        claude_dir: Path | str = DEFAULT_CLAUDE_DIR,
        token_reader: Callable[[], str] | None = None,
        base_url: str = SUBSCRIPTION_BASE_URL,
        provider_id: str = PROVENANCE_SOURCE,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
        transport: httpx.BaseTransport | None = None,
        capture_hook: Callable[..., Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Parent validates base_url/timeouts/etc.; we strip its env-key auth.
        kwargs.pop("env_key", None)
        super().__init__(
            base_url=base_url.rstrip("/") + "/v1",
            env_key="_PPMLX_SUBSCRIPTION_UNUSED",
            provider_id=provider_id,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            transport=transport,
            **kwargs,
        )
        self._parent_base_url = self._base_url
        # Tunnel origin always points at the real API root (no /v1 suffix).
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("Provider base URL is invalid")
        self._tunnel_origin = base_url.rstrip("/")
        self._claude_dir = Path(claude_dir)
        self._token_reader = token_reader or (
            lambda: read_claude_code_oauth_token(self._claude_dir)
        )
        self.capture_hook = capture_hook

    # ------------------------------------------------------------------
    # Credential resolution (live-read, never stored)
    # ------------------------------------------------------------------

    def _resolve_api_key(self) -> str:
        return self._token_reader()

    def _headers(self) -> dict[str, str]:
        # The resolved token lives only in this short-lived header mapping.
        # It is never logged, embedded in exceptions, or exposed via reprs.
        return {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
        }

    def capabilities(self, model_id: str) -> ProviderCapabilities:
        caps = super().capabilities(model_id)
        return ProviderCapabilities(
            text=caps.text,
            images=caps.images,
            tools=caps.tools,
            parallel_tool_calls=caps.parallel_tool_calls,
            reasoning=caps.reasoning,
            streaming=caps.streaming,
            context_window=caps.context_window,
            data_path=ProviderDataPath.REMOTE,
            credential_types=(ProviderCredentialType.OAUTH_SESSION,),
            tool_support_status=caps.tool_support_status,
        )

    def health(self) -> ProviderHealth:
        try:
            self._token_reader()
        except ProviderError:
            return ProviderHealth(
                provider_id=self.provider_id,
                status=ProviderHealthStatus.UNAVAILABLE,
                code="credentials_unavailable",
                model_count=len(self._model_catalog),
            )
        return ProviderHealth(
            provider_id=self.provider_id,
            status=ProviderHealthStatus.HEALTHY,
            code="ready",
            model_count=len(self._model_catalog),
        )

    # ------------------------------------------------------------------
    # Raw full tunnel
    # ------------------------------------------------------------------

    def _tunnel_headers(
        self, incoming: Mapping[str, str] | None
    ) -> dict[str, str]:
        headers = {
            name: value
            for name, value in (incoming or {}).items()
            if name.lower() not in _TUNNEL_DROP_HEADERS
            and name.lower() != "authorization"
        }
        headers["Authorization"] = f"Bearer {self._resolve_api_key()}"
        return headers

    def forward(
        self,
        path: str,
        *,
        method: str = "POST",
        headers: Mapping[str, str] | None = None,
        body: bytes = b"",
        cancel_handle: ProviderCancellationHandle | None = None,
        max_response_bytes: int | None = None,
    ) -> TunnelResponse:
        """Send one raw request to api.anthropic.com and stream the reply.

        Preserves every original end-to-end header and the body byte-for-byte;
        the response is yielded chunk-for-chunk (SSE passthrough). Only
        same-origin paths under ``/v1/`` are tunneled. Raises typed safe
        errors; the token never appears in any failure surface.
        """
        if type(path) is not str or not path.startswith("/v1/") or any(
            character.isspace() for character in path
        ):
            raise ProviderError(
                provider_id=self.provider_id, code="invalid_tunnel_path"
            )
        if method not in ("POST", "GET"):
            raise ProviderError(provider_id=self.provider_id, code="invalid_tunnel_method")
        limit = (
            self._max_response_bytes
            if max_response_bytes is None
            else max_response_bytes
        )
        if type(limit) is not int or limit < 1:
            raise ProviderError(provider_id=self.provider_id, code="invalid_request")

        request = httpx.Request(
            method,
            self._tunnel_origin + path,
            content=body if isinstance(body, (bytes, bytearray)) else bytes(body),
            headers=self._tunnel_headers(headers),
        )
        client = httpx.Client(
            timeout=self._timeout_seconds, transport=self._transport
        )
        try:
            response = client.send(request, stream=True)
        except httpx.TimeoutException:
            client.close()
            raise ProviderError(provider_id=self.provider_id, code="timeout") from None
        except httpx.TransportError:
            client.close()
            raise ProviderError(
                provider_id=self.provider_id, code="network_error"
            ) from None
        except Exception:
            client.close()
            raise ProviderError(
                provider_id=self.provider_id, code="provider_invoke_failed"
            ) from None

        def _chunks() -> Iterator[bytes]:
            total = 0
            try:
                if response.is_stream_consumed:
                    # Non-streaming transport (e.g. buffered mock): content is
                    # already materialized.
                    chunks: Iterator[bytes] = iter([response.content])
                else:
                    chunks = response.iter_raw()
                for chunk in chunks:
                    total += len(chunk)
                    if total > limit:
                        raise ProviderError(
                            provider_id=self.provider_id, code="response_too_large"
                        )
                    if cancel_handle is not None and cancel_handle.cancelled:
                        raise ProviderCancelledError(provider_id=self.provider_id)
                    if chunk:
                        yield chunk
                if cancel_handle is not None and cancel_handle.cancelled:
                    raise ProviderCancelledError(provider_id=self.provider_id)
            finally:
                response.close()
                client.close()

        return TunnelResponse(
            status_code=response.status_code,
            headers={
                name: value
                for name, value in response.headers.items()
                if name.lower()
                not in ("content-length", "connection", "transfer-encoding")
            },
            chunks=_chunks(),
        )

    # ------------------------------------------------------------------
    # Memory capture hook (provenance: anthropic-subscription)
    # ------------------------------------------------------------------

    def capture_to_memory(
        self,
        *,
        request_id: str,
        model_alias: str,
        request_text: str,
        response_text: str | None,
    ) -> None:
        """Feed one full request/response pair into the memory pipeline.

        Standard secret redaction (``auth.redact``) is applied BEFORE the
        content leaves this method; the memory store applies a second
        redaction pass at persistence time.
        """
        if self.capture_hook is None:
            return
        from ppmlx.auth import redact

        try:
            from ppmlx.memory_store import _redact_persisted_value
        except Exception:  # pragma: no cover - store always available in app

            from typing import Any as _Any

            def _redact_persisted_value(
                value: _Any, *, field: str | None = None
            ) -> _Any:
                return value

        payload = {
            "request_id": request_id,
            "endpoint": "/anthropic/v1/messages",
            "model_alias": model_alias,
            "model_repo": PROVENANCE_SOURCE,
            "request_text": str(
                _redact_persisted_value(redact(request_text))
            ),
            "response_text": str(
                _redact_persisted_value(redact(response_text or ""))
            ),
            "metadata": {"source": PROVENANCE_SOURCE},
        }
        self.capture_hook(**payload)


class LockedSubscriptionPassthrough:
    """Stand-in registered when the [dangerous] gate is OFF.

    Routes targeting the subscription passthrough fail with a typed error
    that names the config flag instead of an opaque routing failure.
    """

    def __init__(self, provider_id: str = PROVENANCE_SOURCE) -> None:
        self._provider_id = provider_id

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @staticmethod
    def _error() -> ProviderError:
        return ProviderError(
            provider_id=PROVENANCE_SOURCE,
            code="subscription_passthrough_disabled",
            detail=(
                "set [dangerous] subscription_passthrough = true in "
                "~/.ppmlx/config.toml to enable subscription passthrough"
            ),
        )

    def list_models(self) -> tuple[Any, ...]:
        return ()

    def capabilities(self, model_id: str) -> ProviderCapabilities:
        raise self._error()

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            provider_id=self._provider_id,
            status=ProviderHealthStatus.UNAVAILABLE,
            code="subscription_passthrough_disabled",
            model_count=0,
        )

    def invoke(self, invocation: Any) -> Any:
        raise self._error()

    def stream(self, invocation: Any) -> Iterator[Any]:
        raise self._error()


__all__ = [
    "LockedSubscriptionPassthrough",
    "PROVENANCE_SOURCE",
    "SUBSCRIPTION_BASE_URL",
    "SubscriptionCredentialsUnavailable",
    "SubscriptionPassthroughProvider",
    "TOS_WARNING",
    "TunnelResponse",
    "read_claude_code_oauth_token",
    "warn_subscription_tos_once",
]

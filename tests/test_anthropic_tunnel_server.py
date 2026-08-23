"""Server-level tests: the /anthropic tunnel must not mask typed errors."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import ppmlx.server as server_mod
from ppmlx.providers.base import ProviderError


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(
        server_mod, "_subscription_passthrough_enabled", lambda: True
    )
    from ppmlx.server import app

    return TestClient(app)


def _patch_forward(monkeypatch, exc: Exception) -> None:
    from ppmlx.providers.anthropic_subscription import (
        SubscriptionPassthroughProvider,
    )

    def raise_exc(self, *args, **kwargs):
        raise exc

    monkeypatch.setattr(SubscriptionPassthroughProvider, "forward", raise_exc)


def test_bad_path_returns_404_not_502(client, monkeypatch) -> None:
    # provider.forward raises invalid_tunnel_path for non-/v1/ paths; but we
    # simulate directly to pin the mapping.
    _patch_forward(
        monkeypatch,
        ProviderError(provider_id="anthropic-subscription",
                      code="invalid_tunnel_path"),
    )
    resp = client.post("/anthropic/admin/secret", json={})
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"]["code"] == "invalid_tunnel_path"


def test_real_bad_path_is_404(client) -> None:
    resp = client.post("/anthropic/not-v1/messages", json={"model": "x"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "invalid_tunnel_path"
    assert "tunnel_failed" not in resp.text


def test_bad_method_is_4xx_not_502(client) -> None:
    resp = client.delete("/anthropic/v1/messages")
    # FastAPI's own 405 (no DELETE route) — either way it is a 4xx, not a
    # masked 502 from the tunnel.
    assert resp.status_code == 405
    assert "tunnel_failed" not in resp.text


def test_credentials_unavailable_propagates(client, monkeypatch) -> None:
    from ppmlx.providers.anthropic_subscription import (
        SubscriptionCredentialsUnavailable,
    )

    _patch_forward(
        monkeypatch,
        SubscriptionCredentialsUnavailable(reason="no credentials file"),
    )
    resp = client.post("/anthropic/v1/messages", json={"model": "m"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["error"]["code"] == "credentials_unavailable"
    assert "Claude Code" in body["error"]["message"]
    assert "Keychain" in body["error"]["message"]
    assert "tunnel_failed" not in resp.text


def test_other_provider_errors_keep_code_with_502(
    client, monkeypatch
) -> None:
    _patch_forward(
        monkeypatch,
        ProviderError(provider_id="anthropic-subscription", code="timeout"),
    )
    resp = client.post("/anthropic/v1/messages", json={"model": "m"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "timeout"


def test_unexpected_exception_still_generic_502(client, monkeypatch) -> None:
    _patch_forward(monkeypatch, RuntimeError("boom"))
    resp = client.post("/anthropic/v1/messages", json={"model": "m"})
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "tunnel_failed"


def test_disabled_flag_gets_explicit_403(monkeypatch) -> None:
    monkeypatch.setattr(
        server_mod, "_subscription_passthrough_enabled", lambda: False
    )
    from ppmlx.server import app

    client = TestClient(app)
    resp = client.post("/anthropic/v1/messages", json={"model": "m"})
    body = resp.json()
    assert resp.status_code == 403
    assert body["error"]["code"] == "subscription_passthrough_disabled"
    # The message must name the exact config knob to flip (B1 regression).
    assert "[dangerous] subscription_passthrough" in body["error"]["message"]
    assert "config.toml" in body["error"]["message"]

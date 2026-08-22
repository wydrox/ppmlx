"""Tests for the deterministic route policy and fallback matrix."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from ppmlx.providers.base import ProviderDataPath
from ppmlx.router import (
    FORBIDDEN_FALLBACK_CATEGORIES,
    HealthSnapshot,
    HealthState,
    RequiredCapabilities,
    RouteCandidate,
    RouteEntry,
    RouteInput,
    SnapshotCandidateState,
    StubHealthSource,
    check_capabilities,
    fallback_allowed,
    forbidden_by_data_path,
    load_policy,
    policy_from_dict,
    resolve_alias,
    route,
)


NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=UTC)

POLICY_DICT = {
    "routes": {
        "version": "test-1",
        "default_model": "mlx/main",
        "aliases": {
            "gpt-local": ["mlx", "llama3"],
            "gpt-remote": ["openai", "gpt-5"],
        },
        "entries": [
            {
                "key": "codex:gpt-5",
                "fallback_errors": ["connection", "timeout", "server", "unavailable"],
                "candidates": [
                    {
                        "provider": "openai",
                        "model": "gpt-5",
                        "capability_profile": "default",
                    },
                    {
                        "provider": "mlx",
                        "model": "llama3",
                        "capability_profile": "default",
                    },
                ],
            }
        ],
    }
}

CAPABILITY_MATRIX = {
    # Remote candidate: text-only in this slice.
    ("openai", "gpt-5"): dict(
        local=False, tools=False, parallel_tool_calls=False
    ),
    ("mlx", "llama3"): dict(local=True, tools=True, parallel_tool_calls=True),
    ("mlx", "main"): dict(local=True, tools=True, parallel_tool_calls=True),
}


def _capabilities(candidate: RouteCandidate):
    from ppmlx.providers.base import (
        ProviderCapabilities,
        ProviderCredentialType,
        ProviderStreamingMode,
        ProviderToolSupportStatus,
    )

    spec = CAPABILITY_MATRIX.get(
        (candidate.provider_id, candidate.model), {}
    )
    local = spec.get("local", candidate.provider_id in {"mlx", "local"})
    tools = spec.get("tools", local)
    parallel = spec.get("parallel_tool_calls", tools and local)
    return ProviderCapabilities(
        text=True,
        images=False,
        tools=tools,
        parallel_tool_calls=parallel,
        reasoning=False,
        streaming=ProviderStreamingMode.BUFFERED,
        context_window=spec.get("context_window"),
        data_path=(
            ProviderDataPath.LOCAL if local else ProviderDataPath.REMOTE
        ),
        credential_types=(
            (ProviderCredentialType.NONE,)
            if local
            else (ProviderCredentialType.API_KEY,)
        ),
        tool_support_status=(
            ProviderToolSupportStatus.STABLE
            if tools
            else ProviderToolSupportStatus.DISABLED
        ),
    )


@pytest.fixture(autouse=True)
def _patch_stub_capability_lookup(monkeypatch):
    """Give the router a deterministic capability table per provider model."""
    from ppmlx import router as router_module

    monkeypatch.setattr(router_module, "_stub_capabilities", _capabilities)


def _route_input(**overrides) -> RouteInput:
    values = {
        "public_model": "gpt-5",
        "harness": "codex",
        "harness_version": "0.147.0",
        "protocol": "openai-responses",
        "required": RequiredCapabilities(text=True),
        "policy_version": "test-1",
        "health_snapshot_id": "snap-test",
        "request_id": "req-1",
        "session_id": "sess-1",
    }
    values.update(overrides)
    return RouteInput(**values)


def _snapshot(
    states,
    *,
    captured_at: datetime = NOW,
    ttl_ms: int = 30_000,
) -> HealthSnapshot:
    return HealthSnapshot(
        snapshot_id="snap-test",
        captured_at=captured_at,
        expires_at=captured_at + timedelta(milliseconds=ttl_ms),
        states=MappingProxyType(dict(states)),
    )


def _healthy(candidate: RouteCandidate) -> SnapshotCandidateState:
    return SnapshotCandidateState(
        provider_id=candidate.provider_id,
        model=candidate.model,
        auth_profile=candidate.auth_profile,
        state=HealthState.HEALTHY,
        reason_category="probe_ok",
        checked_at=captured_at_for(candidate),
    )


def captured_at_for(candidate: RouteCandidate) -> datetime:
    del candidate
    return NOW


def test_route_decision_is_deterministic() -> None:
    policy = policy_from_dict(POLICY_DICT)
    snapshot = StubHealthSource().snapshot(policy.entries["codex:gpt-5"].candidates, now=NOW)
    first = route(_route_input(), policy, snapshot=snapshot, now=NOW)
    second = route(_route_input(), policy, snapshot=snapshot, now=NOW)
    assert first == second
    assert first.decision_id == second.decision_id
    assert first.selected == RouteCandidate(provider_id="openai", model="gpt-5")


def test_explicit_request_beats_default_and_default_used_when_missing() -> None:
    policy = policy_from_dict(POLICY_DICT)
    decision = route(_route_input(), policy, now=NOW)
    assert decision.status == "routed"
    assert decision.selected == RouteCandidate(provider_id="openai", model="gpt-5")

    fallback_input = _route_input(public_model="missing-model")
    missing = route(fallback_input, policy, now=NOW)
    assert missing.status == "routed"
    assert missing.selected == RouteCandidate(provider_id="mlx", model="main")


def test_resolve_alias_maps_and_unknown_returns_none() -> None:
    policy = policy_from_dict(POLICY_DICT)
    assert resolve_alias(policy, "gpt-local") == ("mlx", "llama3")
    assert resolve_alias(policy, "nope") is None


def test_capability_rejection_selects_next_candidate() -> None:
    policy = policy_from_dict(POLICY_DICT)
    decision = route(
        _route_input(
            required=RequiredCapabilities(tools=True, parallel_tool_calls=True)
        ),
        policy,
        now=NOW,
    )
    assert decision.status == "routed"
    assert decision.selected == RouteCandidate(provider_id="mlx", model="llama3")
    assert any(
        kind == "capability" and "parallel_tool_calls" in detail
        for _, kind, detail in decision.skipped
    )


def test_all_candidates_fail_capability_status() -> None:
    policy = policy_from_dict(POLICY_DICT)
    decision = route(
        _route_input(required=RequiredCapabilities(images=True)),
        policy,
        now=NOW,
    )
    assert decision.status == "capability"
    assert "images" in decision.missing_capabilities


def test_fallback_matrix_permits_listed_connection_only() -> None:
    policy = policy_from_dict(POLICY_DICT)
    entry = policy.entries["codex:gpt-5"]
    assert fallback_allowed(entry, "connection") is True
    assert fallback_allowed(entry, "timeout") is True
    assert fallback_allowed(entry, "server") is True
    assert fallback_allowed(entry, "unavailable") is True
    # Rate limiting falls back only when the entry lists it explicitly.
    assert fallback_allowed(entry, "rate_limit") is False
    assert fallback_allowed(entry, "auth") is False
    for category in FORBIDDEN_FALLBACK_CATEGORIES:
        assert fallback_allowed(entry, category) is False


def test_data_path_blocks_local_to_remote_but_not_reverse() -> None:
    local = RouteCandidate(provider_id="mlx", model="main")
    remote = RouteCandidate(provider_id="openai", model="gpt-5")
    paths = {
        (local.provider_id, local.model): ProviderDataPath.LOCAL,
        (remote.provider_id, remote.model): ProviderDataPath.REMOTE,
    }
    assert forbidden_by_data_path(local, remote, paths) is True
    assert forbidden_by_data_path(remote, local, paths) is False
    assert forbidden_by_data_path(local, local, paths) is False


def test_expired_snapshot_is_health_expired() -> None:
    policy = policy_from_dict(POLICY_DICT)
    entry = policy.entries["codex:gpt-5"]
    expired = StubHealthSource().snapshot(
        entry.candidates,
        ttl_ms=1_000,
        now=NOW - timedelta(seconds=2),
    )
    decision = route(_route_input(), policy, snapshot=expired, now=NOW)
    assert decision.status == "health_expired"


def test_stale_snapshot_is_health_expired() -> None:
    policy = policy_from_dict(POLICY_DICT)
    entry = policy.entries["codex:gpt-5"]
    stale = StubHealthSource().snapshot(
        entry.candidates,
        ttl_ms=30_000,
        now=NOW - timedelta(milliseconds=30_500),
    )
    decision = route(_route_input(), policy, snapshot=stale, now=NOW)
    assert decision.status == "health_expired"


def test_fresh_snapshot_routes() -> None:
    policy = policy_from_dict(POLICY_DICT)
    entry = policy.entries["codex:gpt-5"]
    fresh = StubHealthSource().snapshot(
        entry.candidates, now=NOW - timedelta(milliseconds=50)
    )
    decision = route(_route_input(), policy, snapshot=fresh, now=NOW)
    assert decision.status == "routed"


def test_duplicate_candidate_tuple_raises_value_error() -> None:
    with pytest.raises(ValueError):
        RouteEntry(
            public_model="m",
            harness="codex",
            candidates=(
                RouteCandidate(provider_id="mlx", model="llama3"),
                RouteCandidate(provider_id="mlx", model="llama3"),
            ),
        )
    with pytest.raises(ValueError):
        RouteEntry(
            public_model="m",
            harness="codex",
            candidates=(
                RouteCandidate(provider_id="mlx", model="llama3"),
                RouteCandidate(
                    provider_id="mlx", model="llama3", auth_profile=None
                ),
            ),
        )


def test_load_policy_reads_tmp_toml(tmp_path) -> None:
    toml_text = """
[routes]
version = "test-1"
default_model = "local/main"

[routes.aliases]
"gpt-local" = ["mlx", "llama3"]

[[routes.entries]]
key = "codex:gpt-5"
fallback_errors = ["connection", "timeout", "server", "unavailable"]

[[routes.entries.candidates]]
provider = "mlx"
model = "llama3"
capability_profile = "default"

[[routes.entries.candidates]]
provider = "openai"
model = "gpt-5"
"""
    path = tmp_path / "routes.toml"
    path.write_text(toml_text, encoding="utf-8")
    policy = load_policy(path)
    assert policy.version == "test-1"
    assert policy.default_model == "local/main"
    assert resolve_alias(policy, "gpt-local") == ("mlx", "llama3")
    entry = policy.entries["codex:gpt-5"]
    assert [candidate.model for candidate in entry.candidates] == ["llama3", "gpt-5"]
    assert fallback_allowed(entry, "connection") is True

"""Audit-fix tests for the deterministic router (M1, M2, L1, L4)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import MappingProxyType

import pytest

from ppmlx.router import (
    HealthSnapshot,
    HealthState,
    RequiredCapabilities,
    RouteCandidate,
    RouteInput,
    SnapshotCandidateState,
    StubHealthSource,
    policy_from_dict,
    resolve_alias,
    route,
)

NOW = datetime(2026, 8, 22, 12, 0, 0, tzinfo=UTC)


def _policy_dict(**routes_extra):
    routes = {
        "version": "test-1",
        "default_model": "mlx/main",
        "aliases": {},
        "entries": [
            {
                "key": "codex:gpt-5",
                "candidates": [{"provider": "mlx", "model": "main"}],
            }
        ],
    }
    routes.update(routes_extra)
    return {"routes": routes}


def _route_input() -> RouteInput:
    return RouteInput(
        public_model="gpt-5",
        harness="codex",
        harness_version="0.147.0",
        protocol="openai-responses",
        required=RequiredCapabilities(text=True),
        policy_version="test-1",
        health_snapshot_id="snap-test",
        request_id="req-1",
        session_id="sess-1",
    )


# ----------------------------------------------------------------------
# M1: snapshot keys include auth_profile
# ----------------------------------------------------------------------


def test_m1_snapshot_keys_include_auth_profile_and_reject_shadowing() -> None:
    moment = NOW
    candidates = (
        RouteCandidate(provider_id="openai", model="gpt-5", auth_profile="work"),
        RouteCandidate(provider_id="openai", model="gpt-5", auth_profile="personal"),
    )
    snapshot = StubHealthSource().snapshot(candidates, now=moment)
    assert ("openai", "work", "gpt-5") in snapshot.states
    assert ("openai", "personal", "gpt-5") in snapshot.states
    # Distinct profiles are distinct keys: no shadowing.
    assert len(snapshot.states) == 2


def test_m1_inconsistent_snapshot_state_key_is_rejected() -> None:
    state = SnapshotCandidateState(
        provider_id="openai",
        model="gpt-5",
        auth_profile="work",
        state=HealthState.HEALTHY,
        reason_category="probe_ok",
        checked_at=NOW,
    )
    with pytest.raises(ValueError):
        HealthSnapshot(
            snapshot_id="snap-x",
            captured_at=NOW,
            expires_at=NOW + timedelta(seconds=30),
            states=MappingProxyType({("openai", "", "gpt-5"): state}),
        )


# ----------------------------------------------------------------------
# M2: injectable capability lookup
# ----------------------------------------------------------------------


def test_m2_capability_lookup_is_injectable() -> None:
    from ppmlx.providers.base import (
        ProviderCapabilities,
        ProviderCredentialType,
        ProviderDataPath,
        ProviderStreamingMode,
    )

    calls: list[RouteCandidate] = []

    def lookup(candidate: RouteCandidate) -> ProviderCapabilities:
        calls.append(candidate)
        return ProviderCapabilities(
            text=False,
            images=False,
            tools=False,
            parallel_tool_calls=False,
            reasoning=False,
            streaming=ProviderStreamingMode.BUFFERED,
            context_window=None,
            data_path=ProviderDataPath.LOCAL,
            credential_types=(ProviderCredentialType.NONE,),
        )

    policy = policy_from_dict(_policy_dict())
    decision = route(
        _route_input(), policy, now=NOW, capability_lookup=lookup
    )
    assert decision.status == "capability"
    assert len(calls) == 1 and calls[0].provider_id == "mlx"


def test_m2_no_module_global_cache_remains() -> None:
    import ppmlx.router as router_module

    assert not hasattr(router_module, "_STUB_CAPABILITY_CACHE")
    assert not hasattr(router_module, "_CAPABILITY_NAMES")
    # Default stub kept for backwards compatibility.
    assert callable(router_module._stub_capabilities)


# ----------------------------------------------------------------------
# L4: duplicate entries and malformed default_model
# ----------------------------------------------------------------------


def test_l4_duplicate_route_entries_are_rejected() -> None:
    mapping = _policy_dict(
        entries=[
            {
                "key": "codex:gpt-5",
                "candidates": [{"provider": "mlx", "model": "main"}],
            },
            {
                "key": "codex:gpt-5",
                "candidates": [{"provider": "openai", "model": "gpt-5"}],
            },
        ]
    )
    with pytest.raises(ValueError, match="duplicate"):
        policy_from_dict(mapping)


def test_l4_malformed_default_model_is_rejected() -> None:
    for bad in ("no-separator", "/missing-provider", "missing-model/"):
        mapping = _policy_dict(default_model=bad)
        with pytest.raises(ValueError, match="default model"):
            policy_from_dict(mapping)


# ----------------------------------------------------------------------
# Alias vs entry shadowing (existing behavior, pinned by test)
# ----------------------------------------------------------------------


def test_alias_shadowed_by_entry_resolves_to_none() -> None:
    mapping = _policy_dict(
        aliases={"codex:gpt-5": ["openai", "other"]},
    )
    policy = policy_from_dict(mapping)
    # Exact route entry key always wins over the alias table.
    assert resolve_alias(policy, "codex:gpt-5") is None

"""Deterministic, capability-aware routing and fallback policy.

The router resolves one public model name to one provider candidate from a
versioned route policy (ADR 0005). Every decision is a pure function of the
route input, the policy snapshot, and one immutable health snapshot; no wall
clock is read unless the caller passes ``now``. Fallback is opt-in per route
entry and never crosses the forbidden error categories or a local-to-remote
data-path boundary.
"""
from __future__ import annotations

import hashlib
import os
import re
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from types import MappingProxyType

from ppmlx.providers.base import (
    ProviderCapabilities,
    ProviderDataPath,
)

_MAX_SNAPSHOT_AGE_MS = 30_000


def _utc(value: datetime) -> datetime:
    """Return an equivalent timezone-aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class HealthState(str, Enum):
    """Sanitized per-candidate health used by candidate selection."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")


@dataclass(frozen=True, slots=True)
class RequiredCapabilities:
    """Capability floor taken from the request and harness contract."""

    text: bool = True
    images: bool = False
    tools: bool = False
    parallel_tool_calls: bool = False
    structured_output: bool = False
    min_context_tokens: int | None = None

    def __post_init__(self) -> None:
        flags = (
            self.text,
            self.images,
            self.tools,
            self.parallel_tool_calls,
            self.structured_output,
        )
        if any(type(value) is not bool for value in flags):
            raise ValueError("Required capability flags are invalid")
        if self.min_context_tokens is not None and (
            type(self.min_context_tokens) is not int or self.min_context_tokens < 1
        ):
            raise ValueError("Required minimum context tokens are invalid")


@dataclass(frozen=True, slots=True)
class RouteCandidate:
    """One ordered route target naming provider, upstream model, and auth."""

    provider_id: str
    model: str
    auth_profile: str | None = None
    capability_profile: str = "default"

    def __post_init__(self) -> None:
        if type(self.provider_id) is not str or not self.provider_id:
            raise ValueError("Route candidate provider identifier is invalid")
        if type(self.model) is not str or not self.model:
            raise ValueError("Route candidate model identifier is invalid")
        if self.auth_profile is not None and (
            type(self.auth_profile) is not str or not self.auth_profile
        ):
            raise ValueError("Route candidate authentication profile is invalid")
        if type(self.capability_profile) is not str or not self.capability_profile:
            raise ValueError("Route candidate capability profile is invalid")


@dataclass(frozen=True, slots=True)
class RouteEntry:
    """Ordered candidate list for one harness and public model pair.

    Candidate tuples must be unique inside an entry so a fallback chain can
    never loop over the same (provider, auth profile, model) target.
    """

    public_model: str
    harness: str
    candidates: tuple[RouteCandidate, ...]
    fallback_permitted_errors: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for name in (self.public_model, self.harness):
            if type(name) is not str or not name:
                raise ValueError("Route entry key parts are invalid")
        if (
            type(self.candidates) is not tuple
            or not self.candidates
            or any(not isinstance(value, RouteCandidate) for value in self.candidates)
        ):
            raise ValueError("Route entry candidates are invalid")
        seen: set[tuple[str, str | None, str]] = set()
        for candidate in self.candidates:
            identity = (candidate.provider_id, candidate.auth_profile, candidate.model)
            if identity in seen:
                raise ValueError("Route entry candidates contain a duplicate target")
            seen.add(identity)
        if type(self.fallback_permitted_errors) is not frozenset or any(
            type(value) is not str or not value
            for value in self.fallback_permitted_errors
        ):
            raise ValueError("Route entry permitted fallback errors are invalid")

    @property
    def key(self) -> str:
        return f"{self.harness}:{self.public_model}"


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """Versioned snapshot of routes, defaults, aliases, and health bounds."""

    version: str
    entries: Mapping[str, RouteEntry]
    default_model: str | None = None
    aliases: Mapping[str, tuple[str, str]] = field(
        default_factory=lambda: MappingProxyType({})
    )
    health_snapshot_max_age_ms: int = _MAX_SNAPSHOT_AGE_MS
    total_attempt_limit: int = 1

    def __post_init__(self) -> None:
        if type(self.version) is not str or not self.version:
            raise ValueError("Route policy version is invalid")
        if not isinstance(self.entries, Mapping) or any(
            type(key) is not str or not isinstance(value, RouteEntry)
            for key, value in self.entries.items()
        ):
            raise ValueError("Route policy entries are invalid")
        for key, entry in self.entries.items():
            if key != f"{entry.harness}:{entry.public_model}":
                raise ValueError("Route policy entry key does not match its entry")
        if self.default_model is not None and (
            type(self.default_model) is not str or not self.default_model
        ):
            raise ValueError("Route policy default model is invalid")
        if not isinstance(self.aliases, Mapping) or any(
            type(alias) is not str
            or not alias
            or type(target) is not tuple
            or len(target) != 2
            or any(type(part) is not str or not part for part in target)
            for alias, target in self.aliases.items()
        ):
            raise ValueError("Route policy aliases are invalid")
        if (
            type(self.health_snapshot_max_age_ms) is not int
            or self.health_snapshot_max_age_ms <= 0
            or self.health_snapshot_max_age_ms > _MAX_SNAPSHOT_AGE_MS
        ):
            raise ValueError("Route policy snapshot age limit is invalid")
        if type(self.total_attempt_limit) is not int or self.total_attempt_limit < 1:
            raise ValueError("Route policy total attempt limit is invalid")
        object.__setattr__(
            self, "entries", MappingProxyType(dict(self.entries))
        )
        object.__setattr__(
            self, "aliases", MappingProxyType(dict(self.aliases))
        )


def policy_from_dict(mapping: Mapping[str, object]) -> RoutePolicy:
    """Build a :class:`RoutePolicy` from a parsed TOML-shaped mapping."""
    if not isinstance(mapping, Mapping):
        raise ValueError("Route policy document is invalid")
    routes = mapping.get("routes")
    if not isinstance(routes, Mapping):
        raise ValueError("Route policy document has no routes section")
    version = routes.get("version")
    if type(version) is not str or not version:
        raise ValueError("Route policy version is invalid")
    default_model = routes.get("default_model")
    if default_model is not None and (
        type(default_model) is not str
        or not default_model
        or "/" not in default_model
        or any(
            not part
            for part in default_model.partition("/")
        )
    ):
        raise ValueError("Route policy default model is invalid")
    raw_aliases = routes.get("aliases", {})
    if not isinstance(raw_aliases, Mapping):
        raise ValueError("Route policy aliases are invalid")
    aliases: dict[str, tuple[str, str]] = {}
    for alias, target in raw_aliases.items():
        if type(alias) is not str or not alias:
            raise ValueError("Route policy alias name is invalid")
        if (
            type(target) is not list
            or len(target) != 2
            or any(type(part) is not str or not part for part in target)
        ):
            raise ValueError("Route policy alias target is invalid")
        aliases[alias] = (target[0], target[1])
    entries: dict[str, RouteEntry] = {}
    raw_entries = routes.get("entries", [])
    if type(raw_entries) is not list:
        raise ValueError("Route policy entries are invalid")
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise ValueError("Route policy entry is invalid")
        harness = raw_entry.get("key")
        if type(harness) is not str or ":" not in harness:
            raise ValueError("Route entry key is invalid")
        harness_name, _, public_model = harness.partition(":")
        if not harness_name or not public_model:
            raise ValueError("Route entry key is invalid")
        raw_candidates = raw_entry.get("candidates", [])
        if type(raw_candidates) is not list or not raw_candidates:
            raise ValueError("Route entry candidates are invalid")
        candidates: list[RouteCandidate] = []
        for raw_candidate in raw_candidates:
            if not isinstance(raw_candidate, Mapping):
                raise ValueError("Route entry candidate is invalid")
            provider_id = raw_candidate.get("provider")
            model = raw_candidate.get("model")
            if type(provider_id) is not str or not provider_id:
                raise ValueError("Route candidate provider identifier is invalid")
            if type(model) is not str or not model:
                raise ValueError("Route candidate model identifier is invalid")
            candidates.append(
                RouteCandidate(
                    provider_id=provider_id,
                    model=model,
                    auth_profile=raw_candidate.get("auth_profile"),
                    capability_profile=raw_candidate.get(
                        "capability_profile", "default"
                    ),
                )
            )
        fallback_errors = raw_entry.get("fallback_errors", [])
        if type(fallback_errors) is not list or any(
            type(value) is not str or not value for value in fallback_errors
        ):
            raise ValueError("Route entry permitted fallback errors are invalid")
        entry = RouteEntry(
            public_model=public_model,
            harness=harness_name,
            candidates=tuple(candidates),
            fallback_permitted_errors=frozenset(fallback_errors),
        )
        if entry.key in entries:
            raise ValueError("Route policy contains duplicate route entries")
        entries[entry.key] = entry
    return RoutePolicy(
        version=version,
        entries=MappingProxyType(entries),
        default_model=default_model,
        aliases=MappingProxyType(aliases),
        health_snapshot_max_age_ms=routes.get(
            "health_snapshot_max_age_ms", _MAX_SNAPSHOT_AGE_MS
        ),
        total_attempt_limit=routes.get("total_attempt_limit", 1),
    )


def load_policy(path: str | os.PathLike[str]) -> RoutePolicy:
    """Load and validate a route policy from a TOML file."""
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError("Route policy document could not be read") from error
    return policy_from_dict(data)


@dataclass(frozen=True, slots=True)
class RouteInput:
    """Correlated request values that fully determine one route decision."""

    public_model: str
    harness: str
    harness_version: str
    protocol: str
    required: RequiredCapabilities
    policy_version: str
    health_snapshot_id: str
    request_id: str
    session_id: str

    def __post_init__(self) -> None:
        scalars = (
            self.public_model,
            self.harness,
            self.harness_version,
            self.protocol,
            self.policy_version,
            self.health_snapshot_id,
            self.request_id,
            self.session_id,
        )
        if any(type(value) is not str or not value for value in scalars):
            raise ValueError("Route input values are invalid")
        if not isinstance(self.required, RequiredCapabilities):
            raise ValueError("Route input required capabilities are invalid")


@dataclass(frozen=True, slots=True)
class SnapshotCandidateState:
    """One sanitized health state recorded by a health source."""

    provider_id: str
    model: str
    auth_profile: str | None
    state: HealthState
    reason_category: str
    checked_at: datetime

    def __post_init__(self) -> None:
        if type(self.provider_id) is not str or not self.provider_id:
            raise ValueError("Snapshot state provider identifier is invalid")
        if type(self.model) is not str or not self.model:
            raise ValueError("Snapshot state model identifier is invalid")
        if self.auth_profile is not None and (
            type(self.auth_profile) is not str or not self.auth_profile
        ):
            raise ValueError("Snapshot state authentication profile is invalid")
        if not isinstance(self.state, HealthState):
            raise ValueError("Snapshot state value is invalid")
        if type(self.reason_category) is not str or _REASON_CODE_RE.fullmatch(
            self.reason_category
        ) is None:
            raise ValueError("Snapshot state reason category is invalid")
        if not isinstance(self.checked_at, datetime):
            raise ValueError("Snapshot state check time is invalid")
        object.__setattr__(
            self, "checked_at", _utc(self.checked_at)
        )


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    """Immutable health states for one candidate set at capture time."""

    snapshot_id: str
    captured_at: datetime
    expires_at: datetime
    states: Mapping[tuple[str, str, str], SnapshotCandidateState] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if type(self.snapshot_id) is not str or not self.snapshot_id:
            raise ValueError("Health snapshot identifier is invalid")
        if not isinstance(self.captured_at, datetime) or not isinstance(
            self.expires_at, datetime
        ):
            raise ValueError("Health snapshot times are invalid")
        captured = _utc(self.captured_at)
        expires = _utc(self.expires_at)
        object.__setattr__(self, "captured_at", captured)
        object.__setattr__(self, "expires_at", expires)
        if expires <= captured:
            raise ValueError("Health snapshot expiry precedes capture time")
        if not isinstance(self.states, Mapping) or any(
            type(key) is not tuple
            or len(key) != 3
            or any(type(part) is not str for part in key)
            or not isinstance(value, SnapshotCandidateState)
            for key, value in self.states.items()
        ):
            raise ValueError("Health snapshot states are invalid")
        for key, value in self.states.items():
            # M1: a snapshot key must name the same candidate as the state
            # it carries, otherwise one entry silently shadows another.
            expected = (
                value.provider_id,
                value.auth_profile or "",
                value.model,
            )
            if key != expected:
                raise ValueError("Health snapshot state key is inconsistent")
        object.__setattr__(
            self, "states", MappingProxyType(dict(self.states))
        )

    def age_ms(self, at: datetime) -> int:
        """Nonnegative snapshot age in milliseconds at decision time."""
        return max(0, int((_utc(at) - self.captured_at).total_seconds() * 1000))

    def state_for(self, candidate: RouteCandidate) -> HealthState:
        """State recorded for one candidate; unknown when absent."""
        found = self.states.get(
            (candidate.provider_id, candidate.auth_profile or "", candidate.model)
        )
        return HealthState.UNKNOWN if found is None else found.state


class StubHealthSource:
    """Stub provider-health source marking every candidate healthy."""

    def snapshot(
        self,
        candidates: tuple[RouteCandidate, ...],
        *,
        ttl_ms: int = 30_000,
        now: datetime | None = None,
    ) -> HealthSnapshot:
        """Build one deterministic all-healthy snapshot for the candidates."""
        moment = _utc(now) if now is not None else _utc(datetime.now(UTC))
        if type(ttl_ms) is not int or ttl_ms < 1:
            raise ValueError("Health snapshot ttl is invalid")
        digest = hashlib.sha256()
        digest.update(moment.isoformat().encode("utf-8"))
        for candidate in sorted(candidates, key=lambda item: (
            item.provider_id, item.auth_profile or "", item.model
        )):
            digest.update(
                f"\x1f{candidate.provider_id}\x1e{candidate.auth_profile or ''}"
                f"\x1e{candidate.model}".encode("utf-8")
            )
        return HealthSnapshot(
            snapshot_id=f"snap-{digest.hexdigest()[:16]}",
            captured_at=moment,
            expires_at=moment + timedelta(milliseconds=ttl_ms),
            states=MappingProxyType(
                {
                    (
                        candidate.provider_id,
                        candidate.auth_profile or "",
                        candidate.model,
                    ): (
                        SnapshotCandidateState(
                            provider_id=candidate.provider_id,
                            model=candidate.model,
                            auth_profile=candidate.auth_profile,
                            state=HealthState.HEALTHY,
                            reason_category="stub_source",
                            checked_at=moment,
                        )
                    )
                    for candidate in candidates
                }
            ),
        )


def resolve_alias(
    policy: RoutePolicy, name: str
) -> tuple[str, str] | None:
    """Resolve one public name to ``(provider_id, provider_model)``.

    Resolution is pure lookup: an exact route entry key always wins over the
    alias table, and an unknown name returns ``None`` instead of guessing.
    """
    if name in policy.entries:
        return None
    return policy.aliases.get(name)


def check_capabilities(
    required: RequiredCapabilities, caps: ProviderCapabilities
) -> tuple[str, ...]:
    """Sorted names of capabilities the provider model is missing."""
    missing: list[str] = []
    if required.text and not caps.text:
        missing.append("text")
    if required.images and not caps.images:
        missing.append("images")
    if required.tools and not caps.tools:
        missing.append("tools")
    if required.parallel_tool_calls and not caps.parallel_tool_calls:
        missing.append("parallel_tool_calls")
    # Structured output requires a tool-capable model in this slice.
    if required.structured_output and not caps.tools:
        missing.append("structured_output")
    if (
        required.min_context_tokens is not None
        and caps.context_window is not None
        and required.min_context_tokens > caps.context_window
    ):
        missing.append("context")
    return tuple(sorted(missing))


FALLBACK_PERMITTED_CATEGORIES = frozenset(
    {"connection", "timeout", "unavailable", "server"}
)

FORBIDDEN_FALLBACK_CATEGORIES = frozenset(
    {
        "auth",
        "permission",
        "billing",
        "invalid_request",
        "safety_refusal",
        "tool_contract",
        "user_cancelled",
        "missing_capability",
    }
)


def fallback_allowed(entry: RouteEntry, category: str) -> bool:
    """True only when the category is globally permitted, listed on the
    entry, and not forbidden."""
    return (
        category in FALLBACK_PERMITTED_CATEGORIES
        and category not in FORBIDDEN_FALLBACK_CATEGORIES
        and category in entry.fallback_permitted_errors
    )


def forbidden_by_data_path(
    candidate_a: RouteCandidate,
    candidate_b: RouteCandidate,
    data_paths: Mapping[tuple[str, str], ProviderDataPath],
) -> bool:
    """True when falling back from a to b moves local-only content remote."""
    path_a = data_paths.get((candidate_a.provider_id, candidate_a.model))
    path_b = data_paths.get((candidate_b.provider_id, candidate_b.model))
    return path_a is ProviderDataPath.LOCAL and path_b is ProviderDataPath.REMOTE


_DECISION_STATUSES = frozenset(
    {"routed", "no_route", "unhealthy", "capability", "health_expired"}
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Typed outcome of one deterministic route selection."""

    decision_id: str
    route_input: RouteInput
    selected: RouteCandidate | None
    pinned_candidate_id: str | None
    skipped: tuple[tuple[RouteCandidate, str, str], ...]
    status: str
    missing_capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.decision_id) is not str or not self.decision_id:
            raise ValueError("Route decision identifier is invalid")
        if not isinstance(self.route_input, RouteInput):
            raise ValueError("Route decision input is invalid")
        if self.selected is not None and not isinstance(
            self.selected, RouteCandidate
        ):
            raise ValueError("Route decision selection is invalid")
        if self.pinned_candidate_id is not None and (
            type(self.pinned_candidate_id) is not str
            or not self.pinned_candidate_id
        ):
            raise ValueError("Route decision pinned candidate is invalid")
        if (
            type(self.skipped) is not tuple
            or any(
                type(skip) is not tuple
                or len(skip) != 3
                or not isinstance(skip[0], RouteCandidate)
                or type(skip[1]) is not str
                or not skip[1]
                or type(skip[2]) is not str
                or not skip[2]
                for skip in self.skipped
            )
        ):
            raise ValueError("Route decision skip records are invalid")
        if self.status not in _DECISION_STATUSES:
            raise ValueError("Route decision status is invalid")
        if type(self.missing_capabilities) is not tuple or any(
            type(name) is not str or not name
            for name in self.missing_capabilities
        ):
            raise ValueError("Route decision missing capabilities are invalid")
        object.__setattr__(
            self, "skipped", tuple(self.skipped)
        )


def route(
    request: RouteInput,
    policy: RoutePolicy,
    snapshot: HealthSnapshot | None = None,
    now: datetime | None = None,
    capability_lookup: Callable[[RouteCandidate], ProviderCapabilities] | None = None,
) -> RouteDecision:
    """Select one candidate deterministically from the policy and snapshot.

    ``capability_lookup`` maps one candidate to its advertised capabilities.
    It defaults to :func:`_stub_capabilities`, a placeholder table kept for
    backwards compatibility until adapters register real profiles; callers
    should inject a real lookup in production.
    """
    moment = _utc(now) if now is not None else _utc(datetime.now(UTC))
    entry = policy.entries.get(f"{request.harness}:{request.public_model}")
    if entry is None:
        entry = _default_entry(policy)
        if entry is None:
            return _decision(request, policy, None, (), "no_route", ())
    if snapshot is None:
        snapshot = StubHealthSource().snapshot(
            entry.candidates, now=moment
        )
    if (
        snapshot.age_ms(moment) > policy.health_snapshot_max_age_ms
        or moment >= snapshot.expires_at
    ):
        return _decision(request, policy, None, (), "health_expired", ())
    if capability_lookup is None:
        capability_lookup = _stub_capabilities
    skipped: list[tuple[RouteCandidate, str, str]] = []
    capability_failures = 0
    missing_all: set[str] = set()
    for candidate in entry.candidates:
        state = snapshot.state_for(candidate)
        if state is not HealthState.HEALTHY:
            skipped.append(
                (candidate, state.value, f"{state.value}_candidate_state")
            )
            continue
        capabilities = capability_lookup(candidate)
        missing = check_capabilities(request.required, capabilities)
        if missing:
            capability_failures += 1
            missing_all.update(missing)
            skipped.append((candidate, "capability", ",".join(missing)))
            continue
        return _decision(request, policy, candidate, tuple(skipped), "routed", ())
    return _decision(
        request,
        policy,
        None,
        tuple(skipped),
        "capability" if capability_failures else "unhealthy",
        tuple(sorted(missing_all)),
    )


def _default_entry(policy: RoutePolicy) -> RouteEntry | None:
    """Synthesize the single-candidate default entry from ``default_model``."""
    if policy.default_model is None:
        return None
    provider_id, separator, model = policy.default_model.partition("/")
    if not separator or not provider_id or not model:
        return None
    candidate = RouteCandidate(provider_id=provider_id, model=model)
    return RouteEntry(
        public_model=policy.default_model.replace("/", "-"),
        harness="default",
        candidates=(candidate,),
    )


def _stub_capabilities(candidate: RouteCandidate) -> ProviderCapabilities:
    """Placeholder capability lookup until adapters register real profiles.

    Local candidates advertise full tool support; remote candidates stay
    text-only so capability tests can drive rejection deterministically.
    """
    from ppmlx.providers.base import (
        ProviderCredentialType,
        ProviderStreamingMode,
        ProviderToolSupportStatus,
    )

    local = candidate.provider_id in {"mlx", "local"}
    capabilities = ProviderCapabilities(
        text=True,
        images=False,
        tools=local,
        parallel_tool_calls=local,
        reasoning=False,
        streaming=ProviderStreamingMode.BUFFERED,
        context_window=None,
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
            if local
            else ProviderToolSupportStatus.DISABLED
        ),
    )
    return capabilities


def _decision(
    request: RouteInput,
    policy: RoutePolicy,
    selected: RouteCandidate | None,
    skipped: tuple[tuple[RouteCandidate, str, str], ...],
    status: str,
    missing_capabilities: tuple[str, ...],
) -> RouteDecision:
    digest = hashlib.sha256()
    digest.update(policy.version.encode("utf-8"))
    for value in (
        request.health_snapshot_id,
        request.harness,
        request.public_model,
        request.request_id,
    ):
        digest.update(b"\x1f")
        digest.update(value.encode("utf-8"))
    return RouteDecision(
        decision_id=digest.hexdigest()[:16],
        route_input=request,
        selected=selected,
        pinned_candidate_id=None,
        skipped=skipped,
        status=status,
        missing_capabilities=missing_capabilities,
    )


__all__ = [
    "FALLBACK_PERMITTED_CATEGORIES",
    "FORBIDDEN_FALLBACK_CATEGORIES",
    "HealthSnapshot",
    "HealthState",
    "RequiredCapabilities",
    "RouteCandidate",
    "RouteDecision",
    "RouteEntry",
    "RouteInput",
    "RoutePolicy",
    "SnapshotCandidateState",
    "StubHealthSource",
    "check_capabilities",
    "fallback_allowed",
    "forbidden_by_data_path",
    "load_policy",
    "policy_from_dict",
    "resolve_alias",
    "route",
]

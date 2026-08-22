"""End-to-end wiring of the router and remote providers (ADR 0005 MVP).

One service turns a :class:`~ppmlx.router.RouteInput` into a
:class:`~ppmlx.router.RouteDecision` and then invokes the selected remote
provider, returning canonical Agent IR events. Capability lookup is injected
from the real providers' ``capabilities()`` (audit fix M2 at this layer);
credentials resolve through ``ppmlx.auth`` keyring storage.

Fallback pinning contract: a failed attempt may fall back to the next
candidate only while zero output events have been emitted. Once the first
output event has been observed, the route is pinned — no provider switch.
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime

from ppmlx.auth import AuthError, resolve_secret
from ppmlx.providers.base import (
    Provider,
    ProviderCancellationHandle,
    ProviderCancelledError,
    ProviderError,
    ProviderInvocation,
)
from ppmlx.router import (
    FORBIDDEN_FALLBACK_CATEGORIES,
    HealthSnapshot,
    HealthState,
    RequiredCapabilities,
    RouteCandidate,
    RouteDecision,
    RouteEntry,
    RouteInput,
    RoutePolicy,
    SnapshotCandidateState,
    StubHealthSource,
    fallback_allowed,
    resolve_alias,
    route as _route,
)

__all__ = [
    "LOCAL_PROVIDER_IDS",
    "RoutedResult",
    "RoutingService",
    "RoutingServiceError",
    "prime_provider_credentials",
]


LOCAL_PROVIDER_IDS = frozenset({"mlx", "local"})

# Provider error code -> router fallback category. Codes absent here are
# never eligible for fallback (they map onto the forbidden matrix).
_ERROR_CATEGORY = {
    "timeout": "timeout",
    "network_error": "connection",
    "connection_error": "connection",
    "unavailable": "unavailable",
    "service_unavailable": "unavailable",
    "rate_limited": "unavailable",
    "server_error": "server",
    "provider_invoke_failed": "server",
}


class RoutingServiceError(Exception):
    """Typed, secret-free failure of one routed request."""

    def __init__(
        self,
        code: str,
        *,
        status_code: int = 503,
        decision: RouteDecision | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code if 400 <= status_code <= 599 else 500
        self.decision = decision
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RoutedResult:
    """Outcome of one successful routed provider invocation."""

    decision: RouteDecision
    events: tuple
    provider_id: str
    model_id: str


def _category_for(code: str) -> str | None:
    return _ERROR_CATEGORY.get(code)


def _snapshot_with_failure(
    base: HealthSnapshot,
    candidate: RouteCandidate,
    *,
    reason_category: str,
) -> HealthSnapshot:
    """Return a copy of ``base`` marking one candidate unhealthy."""
    states = dict(base.states)
    key = (candidate.provider_id, candidate.auth_profile or "", candidate.model)
    states[key] = SnapshotCandidateState(
        provider_id=candidate.provider_id,
        model=candidate.model,
        auth_profile=candidate.auth_profile,
        state=HealthState.UNHEALTHY,
        reason_category=reason_category,
        checked_at=datetime.now(UTC),
    )
    from types import MappingProxyType

    return HealthSnapshot(
        snapshot_id=base.snapshot_id,
        captured_at=base.captured_at,
        expires_at=base.expires_at,
        states=MappingProxyType(states),
    )


def _is_output_event(event) -> bool:
    """True once any content or terminal output has been produced."""
    kind = getattr(event, "type", "")
    return kind in {
        "content.started",
        "content.delta",
        "content.completed",
        "tool_call.started",
        "tool_call.arguments.delta",
        "tool_call.completed",
        "response.refused",
        "response.completed",
        "response.cancelled",
        "response.failed",
    }


class RoutingService:
    """Route one request to a remote provider and collect its Agent IR events."""

    def __init__(
        self,
        policy: RoutePolicy,
        providers: Mapping[str, Provider],
        *,
        health_source=None,
    ) -> None:
        self._policy = policy
        self._providers = dict(providers)
        unknown = set(self._providers) - {p.provider_id for p in self._providers.values()}
        # Mapping keys must match each provider's declared id.
        if unknown:
            raise ValueError("Provider registry keys do not match provider ids")
        self._health_source = health_source if health_source is not None else StubHealthSource()

    @property
    def policy(self) -> RoutePolicy:
        return self._policy

    def remote_alias_target(self, name: str) -> tuple[str, str] | None:
        """Resolve one public alias to ``(provider_id, provider_model)``."""
        return resolve_alias(self._policy, name)

    def is_remote_model(self, name: str) -> bool:
        target = self.remote_alias_target(name)
        return target is not None and target[0] not in LOCAL_PROVIDER_IDS

    def _capability_lookup(self, candidate: RouteCandidate):
        """Real capability lookup injected into the router (audit fix M2)."""
        provider = self._providers.get(candidate.provider_id)
        if provider is None:
            raise RoutingServiceError("unknown_provider", status_code=503)
        try:
            return provider.capabilities(candidate.model)
        except ProviderError:
            raise
        except Exception:
            raise RoutingServiceError(
                "capability_lookup_failed", status_code=503
            ) from None

    def route(
        self,
        request: RouteInput,
        *,
        snapshot: HealthSnapshot | None = None,
        now: datetime | None = None,
    ) -> RouteDecision:
        """One deterministic decision using real provider capabilities."""
        return _route(
            request,
            self._policy,
            snapshot=snapshot,
            now=now,
            capability_lookup=self._capability_lookup,
        )

    def _entry_for(self, request: RouteInput) -> RouteEntry | None:
        entry = self._policy.entries.get(f"{request.harness}:{request.public_model}")
        if entry is not None:
            return entry
        # Default-model entries synthesize public_model with '/'->'-'.
        for candidate_entry in self._policy.entries.values():
            if candidate_entry.harness == request.harness and (
                candidate_entry.public_model.replace("/", "-")
                == request.public_model
            ):
                return candidate_entry
        return None

    def execute(
        self,
        request: RouteInput,
        envelope,
        *,
        max_tokens_cap: int = 32_768,
        cancel_handle: ProviderCancellationHandle | None = None,
        snapshot: HealthSnapshot | None = None,
        now: datetime | None = None,
    ) -> RoutedResult:
        """Buffered execution: route, invoke, fall back only before output."""
        last_decision: RouteDecision | None = None
        attempts = 0
        current_snapshot = snapshot
        while attempts < max(1, self._policy.total_attempt_limit) + len(
            self._providers
        ):
            decision = self.route(request, snapshot=current_snapshot, now=now)
            last_decision = decision
            if decision.status != "routed" or decision.selected is None:
                raise RoutingServiceError(
                    f"route_{decision.status}", status_code=503, decision=decision
                )
            attempts += 1
            candidate = decision.selected
            provider = self._providers.get(candidate.provider_id)
            if provider is None:
                raise RoutingServiceError(
                    "unknown_provider", status_code=503, decision=decision
                )
            invocation = ProviderInvocation(
                request=envelope,
                model_id=candidate.model,
                max_tokens_cap=max_tokens_cap,
                cancel_handle=cancel_handle,
            )
            try:
                result = provider.invoke(invocation)
            except ProviderCancelledError:
                raise
            except ProviderError as error:
                category = _category_for(error.code)
                entry = self._entry_for(request)
                if (
                    category is not None
                    and entry is not None
                    and fallback_allowed(entry, category)
                ):
                    current_snapshot = _snapshot_with_failure(
                        self._health_source.snapshot(
                            entry.candidates, now=now
                        )
                        if current_snapshot is None
                        else current_snapshot,
                        candidate,
                        reason_category=f"{error.code}",
                    )
                    continue
                raise RoutingServiceError(
                    "provider_" + error.code, status_code=502, decision=decision
                ) from error
            return RoutedResult(
                decision=decision,
                events=result.events,
                provider_id=result.provider_id,
                model_id=result.model_id,
            )
        raise RoutingServiceError(
            "route_attempts_exhausted", status_code=503, decision=last_decision
        )

    def stream(
        self,
        request: RouteInput,
        envelope,
        *,
        max_tokens_cap: int = 32_768,
        cancel_handle: ProviderCancellationHandle | None = None,
        now: datetime | None = None,
    ) -> Iterator:
        """Streaming execution with first-event pinning.

        Fallback to later candidates happens only on provider errors raised
        before the first output event; after that the route is pinned and a
        mid-stream failure surfaces as a typed terminal failure event rather
        than a provider switch.
        """
        yielded_any = False
        pinned_provider_id: str | None = None
        snapshot = None
        attempts = 0
        while True:
            decision = self.route(request, snapshot=snapshot, now=now)
            if decision.status != "routed" or decision.selected is None:
                raise RoutingServiceError(
                    f"route_{decision.status}", status_code=503, decision=decision
                )
            attempts += 1
            candidate = decision.selected
            provider = self._providers.get(candidate.provider_id)
            if provider is None:
                raise RoutingServiceError(
                    "unknown_provider", status_code=503, decision=decision
                )
            invocation = ProviderInvocation(
                request=envelope,
                model_id=candidate.model,
                max_tokens_cap=max_tokens_cap,
                cancel_handle=cancel_handle,
            )
            try:
                iterator = provider.stream(invocation)
                while True:
                    try:
                        event = next(iterator)
                    except StopIteration:
                        return
                    yielded_any = True
                    pinned_provider_id = candidate.provider_id
                    yield event
            except ProviderCancelledError:
                raise
            except ProviderError as error:
                if pinned_provider_id is not None or yielded_any:
                    # Pinned: no provider switch after first output event.
                    raise
                category = _category_for(error.code)
                entry = self._entry_for(request)
                if (
                    category is not None
                    and entry is not None
                    and fallback_allowed(entry, category)
                ):
                    snapshot = _snapshot_with_failure(
                        self._health_source.snapshot(entry.candidates, now=now)
                        if snapshot is None
                        else snapshot,
                        candidate,
                        reason_category=f"{error.code}",
                    )
                    if attempts >= self._policy.total_attempt_limit + len(
                        self._providers
                    ) + 1:
                        raise RoutingServiceError(
                            "route_attempts_exhausted",
                            status_code=503,
                            decision=decision,
                        ) from error
                    continue
                raise RoutingServiceError(
                    "provider_" + error.code, status_code=502, decision=decision
                ) from error


def prime_provider_credentials(
    provider_ids: tuple[str, ...] = ("openai", "anthropic"),
    *,
    prefer_env: bool = True,
) -> tuple[str, ...]:
    """Resolve keyring-stored API keys into provider env variables.

    Returns the ids of providers whose credentials were resolved. Values are
    written only to process environment variables read by the provider
    adapters; they are never logged or embedded in errors.
    """
    env_keys = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
    primed: list[str] = []
    for provider_id in provider_ids:
        env_key = env_keys.get(provider_id)
        if env_key is None or os.environ.get(env_key):
            continue
        try:
            _, source = resolve_secret(provider_id, prefer_env=prefer_env)
        except AuthError:
            continue
        if source == "keyring":
            try:
                secret = resolve_secret(provider_id, prefer_env=False)[0]
            except AuthError:
                continue
            os.environ[env_key] = secret
            primed.append(provider_id)
    return tuple(primed)

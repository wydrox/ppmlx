"""In-memory continuation state for harness-owned tool calls."""
from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Callable, Literal, Mapping, Sequence

from ppmlx.agent_ir import new_output_id
from ppmlx.protocols.base import CallReference


DEFAULT_ACTIVE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_TOMBSTONE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_ENTRIES = 16_384

__all__ = [
    "DEFAULT_ACTIVE_TTL_SECONDS",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_TOMBSTONE_TTL_SECONDS",
    "CallConflictError",
    "CallRegistration",
    "CallState",
    "ContinuationExpiredError",
    "ContinuationLedger",
    "ContinuationLedgerError",
    "ContinuationOutcome",
    "ContinuationProbe",
    "ContinuationScope",
    "ContinuationTicket",
    "ConversationMismatchError",
    "IncompleteCallGroupError",
    "InvalidCallStateError",
    "LedgerCapacityError",
    "LedgerKey",
    "LedgerSnapshot",
    "ResultConflictError",
    "ResultIdentity",
    "ResultReceipt",
    "RoutePin",
    "UnknownCallError",
]

_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_ERROR_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class CallState(StrEnum):
    """A tool call state from ADR 0003."""

    STARTED = "started"
    ARGUMENTS_COMPLETE = "arguments_complete"
    WAITING_FOR_RESULT = "waiting_for_result"
    RESULT_RECEIVED = "result_received"
    CONTINUING = "continuing"
    RESOLVED = "resolved"
    ABANDONED = "abandoned"


class ContinuationLedgerError(ValueError):
    """A typed ledger error with a safe public message."""

    code = "continuation_ledger_error"

    def __init__(self) -> None:
        super().__init__(f"Continuation ledger error {self.code}")


class UnknownCallError(ContinuationLedgerError):
    code = "tool_call_unknown"


class ConversationMismatchError(ContinuationLedgerError):
    code = "tool_conversation_mismatch"


class ResultConflictError(ContinuationLedgerError):
    code = "tool_result_conflict"


class ContinuationExpiredError(ContinuationLedgerError):
    code = "tool_continuation_expired"


class InvalidCallStateError(ContinuationLedgerError):
    code = "tool_call_invalid_state"


class CallConflictError(ContinuationLedgerError):
    code = "tool_call_conflict"


class IncompleteCallGroupError(ContinuationLedgerError):
    code = "tool_results_incomplete"


class LedgerCapacityError(ContinuationLedgerError):
    code = "continuation_capacity_exceeded"


@dataclass(frozen=True, slots=True)
class LedgerKey:
    """The complete isolation key for one tool call."""

    principal_id: str
    project_id: str
    harness: str
    conversation_id: str
    call_id: str

    def __post_init__(self) -> None:
        for value in (
            self.principal_id,
            self.project_id,
            self.harness,
            self.conversation_id,
            self.call_id,
        ):
            _require_identifier(value)


@dataclass(frozen=True, slots=True)
class RoutePin:
    """A sanitized route identity for the complete tool round trip."""

    decision_id: str
    provider: str
    model: str
    candidate_id: str

    def __post_init__(self) -> None:
        for value in (self.decision_id, self.provider, self.model, self.candidate_id):
            _require_identifier(value)


@dataclass(frozen=True, slots=True)
class ContinuationScope:
    """The caller scope used before a conversation identifier is known."""

    principal_id: str
    project_id: str
    harness: str

    def __post_init__(self) -> None:
        for value in (self.principal_id, self.project_id, self.harness):
            _require_identifier(value)


@dataclass(frozen=True, slots=True)
class CallRegistration:
    """The stable identity from the first provider tool-call event."""

    key: LedgerKey
    source_call_id: str | None
    tool_name: str
    initial_request_id: str
    choice_index: int
    output_id: str
    tool_call_index: int
    route_pin: RoutePin
    prior_continuation_request_ids: tuple[str, ...] = ()
    parallel_group_id: str | None = None

    def __post_init__(self) -> None:
        if self.source_call_id is not None:
            _require_identifier(self.source_call_id)
        _require_identifier(self.tool_name)
        _require_identifier(self.initial_request_id)
        _require_index(self.choice_index)
        _require_identifier(self.output_id)
        _require_index(self.tool_call_index)
        if not isinstance(self.key, LedgerKey):
            raise ValueError("A call registration key is invalid")
        if not isinstance(self.route_pin, RoutePin):
            raise ValueError("A call route pin is invalid")
        if type(self.prior_continuation_request_ids) is not tuple:
            raise ValueError("Prior continuation request identifiers must be a tuple")
        for request_id in self.prior_continuation_request_ids:
            _require_identifier(request_id)
        request_ids = (self.initial_request_id, *self.prior_continuation_request_ids)
        if len(set(request_ids)) != len(request_ids):
            raise ValueError("Request identifiers must be unique")
        if self.parallel_group_id is not None:
            _require_identifier(self.parallel_group_id)


@dataclass(frozen=True, slots=True)
class ResultIdentity:
    """The identity fields on one harness tool result."""

    request_id: str
    parent_request_id: str
    choice_index: int
    tool_call_index: int
    result_digest: str
    source_output_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.request_id)
        _require_identifier(self.parent_request_id)
        _require_index(self.choice_index)
        _require_index(self.tool_call_index)
        if not _DIGEST.fullmatch(self.result_digest):
            raise ValueError("The result digest must be a lowercase SHA-256 digest")
        if self.source_output_id is not None:
            _require_identifier(self.source_output_id)


@dataclass(frozen=True, slots=True)
class ContinuationOutcome:
    """A safe terminal status for a provider continuation."""

    state: Literal[CallState.RESOLVED, CallState.ABANDONED]
    error_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, CallState) or self.state not in {
            CallState.RESOLVED,
            CallState.ABANDONED,
        }:
            raise ValueError("A continuation outcome must be terminal")
        if self.state is CallState.RESOLVED and self.error_code is not None:
            raise ValueError("A resolved continuation cannot contain an error code")
        if self.error_code is not None and not _SAFE_ERROR_CODE.fullmatch(self.error_code):
            raise ValueError("A continuation error code is invalid")


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """A secret-free immutable view of one ledger entry."""

    key: LedgerKey
    source_call_id: str | None
    state: CallState
    result_digest: str | None
    route_pin: RoutePin
    expires_at: float
    initial_request_id: str
    continuation_request_ids: tuple[str, ...]
    choice_index: int
    output_id: str
    source_result_output_id: str | None
    tool_call_index: int
    parallel_group_id: str | None
    tombstone: bool
    terminal_outcome: ContinuationOutcome | None


@dataclass(frozen=True, slots=True)
class ResultReceipt:
    """The status after the ledger accepts or finds a tool result."""

    disposition: Literal["accepted", "retry"]
    snapshot: LedgerSnapshot


@dataclass(frozen=True, slots=True)
class ContinuationProbe:
    """The adapter context data for one tool-result request."""

    conversation_id: str
    parent_request_id: str
    route_pin: RoutePin
    prior_calls: Mapping[str, CallReference]
    result_output_ids: Mapping[str, str]


class ContinuationTicket:
    """A read-only handle for one shared provider continuation."""

    __slots__ = ("_future", "disposition")

    def __init__(
        self,
        disposition: Literal["owner", "join", "replay"],
        future: Future[ContinuationOutcome],
    ) -> None:
        self.disposition = disposition
        self._future = future

    @property
    def done(self) -> bool:
        return self._future.done()

    def result(self, timeout: float | None = None) -> ContinuationOutcome:
        """Wait for the safe terminal status."""

        return self._future.result(timeout=timeout)

    async def wait(self, timeout: float | None = None) -> ContinuationOutcome:
        """Wait without blocking an asynchronous server worker."""

        wrapped = asyncio.wrap_future(self._future)
        return await asyncio.wait_for(asyncio.shield(wrapped), timeout=timeout)


@dataclass(slots=True)
class _Entry:
    registration: CallRegistration
    state: CallState
    expires_at: float
    result_digest: str | None = None
    result_identity: ResultIdentity | None = None
    continuation_request_ids: tuple[str, ...] = ()
    tombstone: bool = False
    terminal_outcome: ContinuationOutcome | None = None
    flight: Future[ContinuationOutcome] | None = None


class ContinuationLedger:
    """Keep tool-call continuations in process with per-call locking."""

    def __init__(
        self,
        *,
        active_ttl_seconds: float = DEFAULT_ACTIVE_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._active_ttl = _require_ttl(active_ttl_seconds)
        self._tombstone_ttl = float(DEFAULT_TOMBSTONE_TTL_SECONDS)
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("The ledger entry limit must be a positive integer")
        self._max_entries = max_entries
        if not callable(clock):
            raise TypeError("The ledger clock must be callable")
        self._clock = clock
        self._entries: dict[LedgerKey, _Entry] = {}
        self._locks: dict[LedgerKey, threading.RLock] = {}
        self._index: dict[tuple[str, str, str, str, int, str, int], str] = {}
        self._source_index: dict[tuple[str, str, str, str, str, str], str] = {}
        self._scope_index: dict[tuple[str, str, str, str], set[LedgerKey]] = {}
        self._table_lock = threading.RLock()

    def register_call(self, registration: CallRegistration) -> LedgerSnapshot:
        """Register one provider tool call in the started state."""

        if not isinstance(registration, CallRegistration):
            raise TypeError("registration must be a CallRegistration")
        key = registration.key
        with self._table_lock:
            existing = self._entries.get(key)
            if existing is not None:
                lock = self._locks[key]
                with lock:
                    now = self._now()
                    if now >= existing.expires_at:
                        if existing.tombstone:
                            self._remove(key, existing)
                        else:
                            self._expire(existing, now)
                            raise ContinuationExpiredError
                    elif self._is_expired_tombstone(existing):
                        raise ContinuationExpiredError
                    elif existing.registration == registration and existing.state is CallState.STARTED:
                        return self._snapshot(existing)
                    else:
                        raise CallConflictError

            index_key = self._index_key(registration)
            indexed_call = self._index.get(index_key)
            if indexed_call is not None and indexed_call != key.call_id:
                raise CallConflictError
            source_key = self._source_key(registration)
            if source_key is not None:
                indexed_source = self._source_index.get(source_key)
                if indexed_source is not None and indexed_source != key.call_id:
                    raise CallConflictError

            self._make_room_for_entry()

            entry = _Entry(
                registration=registration,
                state=CallState.STARTED,
                expires_at=self._now() + self._active_ttl,
            )
            self._entries[key] = entry
            self._locks[key] = threading.RLock()
            self._index[index_key] = key.call_id
            if source_key is not None:
                self._source_index[source_key] = key.call_id
            self._scope_index.setdefault(self._scope_key(key), set()).add(key)
            return self._snapshot(entry)

    def register_calls(
        self,
        registrations: Sequence[CallRegistration],
    ) -> tuple[LedgerSnapshot, ...]:
        """Register one output group atomically."""

        if isinstance(registrations, (str, bytes)) or not isinstance(registrations, Sequence):
            raise TypeError("registrations must be a sequence")
        items = tuple(registrations)
        if not items:
            raise ValueError("registrations must not be empty")
        if any(not isinstance(item, CallRegistration) for item in items):
            raise TypeError("registrations must contain CallRegistration values")
        if len({item.key for item in items}) != len(items):
            raise CallConflictError
        with self._table_lock:
            added: list[LedgerKey] = []
            snapshots: list[LedgerSnapshot] = []
            try:
                for registration in items:
                    existed = registration.key in self._entries
                    snapshots.append(self.register_call(registration))
                    if not existed:
                        added.append(registration.key)
            except Exception:
                for key in reversed(added):
                    entry = self._entries.get(key)
                    if entry is not None:
                        self._remove(key, entry)
                raise
            return tuple(snapshots)

    def _make_room_for_entry(self) -> None:
        if len(self._entries) < self._max_entries:
            return
        self.cleanup()
        if len(self._entries) >= self._max_entries:
            raise LedgerCapacityError

    def probe_calls(
        self,
        scope: ContinuationScope,
        call_ids: Sequence[str],
        *,
        result_output_ids: Mapping[str, str] | None = None,
    ) -> ContinuationProbe:
        """Resolve calls in one scope without a supplied conversation identifier."""

        if not isinstance(scope, ContinuationScope):
            raise TypeError("scope must be a ContinuationScope")
        if isinstance(call_ids, (str, bytes)) or not isinstance(call_ids, Sequence):
            raise TypeError("call_ids must be a sequence")
        ordered_call_ids = tuple(call_ids)
        if not ordered_call_ids or len(set(ordered_call_ids)) != len(ordered_call_ids):
            raise ConversationMismatchError
        for call_id in ordered_call_ids:
            _require_identifier(call_id)
        try:
            supplied_output_ids = dict(result_output_ids) if result_output_ids is not None else {}
        except Exception:
            raise ConversationMismatchError from None
        if set(supplied_output_ids) - set(ordered_call_ids):
            raise ConversationMismatchError
        for call_id, output_id in supplied_output_ids.items():
            _require_identifier(call_id)
            _require_identifier(output_id)

        with self._table_lock:
            keys: list[LedgerKey] = []
            for call_id in ordered_call_ids:
                matches = self._scope_index.get(
                    (scope.principal_id, scope.project_id, scope.harness, call_id),
                    set(),
                )
                if not matches:
                    raise UnknownCallError
                if len(matches) != 1:
                    raise ConversationMismatchError
                keys.append(next(iter(matches)))

            locks = [self._locks[key] for key in keys]
            for lock in locks:
                lock.acquire()
            try:
                entries = [self._get_locked_entry(key) for key in keys]
                conversation_ids = {entry.registration.key.conversation_id for entry in entries}
                parent_request_ids = {
                    self._call_request_id(entry.registration) for entry in entries
                }
                request_chains = {
                    (
                        entry.registration.initial_request_id,
                        entry.registration.prior_continuation_request_ids,
                    )
                    for entry in entries
                }
                route_pins = {entry.registration.route_pin for entry in entries}
                if (
                    len(conversation_ids) != 1
                    or len(parent_request_ids) != 1
                    or len(request_chains) != 1
                    or len(route_pins) != 1
                ):
                    raise ConversationMismatchError
                if any(
                    entry.state in {CallState.STARTED, CallState.ARGUMENTS_COMPLETE}
                    for entry in entries
                ):
                    raise InvalidCallStateError
                candidate_outputs: dict[str, str] = {}
                for entry in entries:
                    call_id = entry.registration.key.call_id
                    supplied = supplied_output_ids.get(call_id)
                    accepted_output_id = (
                        entry.result_identity.source_output_id
                        if entry.result_identity is not None
                        else None
                    )
                    known_outputs = {
                        output_id
                        for output_id in (supplied, accepted_output_id)
                        if output_id is not None
                    }
                    if len(known_outputs) > 1:
                        raise ConversationMismatchError
                    candidate_outputs[call_id] = (
                        supplied or accepted_output_id or new_output_id()
                    )
                if len(set(candidate_outputs.values())) != len(candidate_outputs):
                    raise ConversationMismatchError
                references = {
                    entry.registration.key.call_id: CallReference(
                        call_id=entry.registration.key.call_id,
                        name=entry.registration.tool_name,
                        choice_index=entry.registration.choice_index,
                        output_id=entry.registration.output_id,
                        tool_call_index=entry.registration.tool_call_index,
                        parallel_group_id=entry.registration.parallel_group_id,
                    )
                    for entry in entries
                }
                return ContinuationProbe(
                    conversation_id=next(iter(conversation_ids)),
                    parent_request_id=next(iter(parent_request_ids)),
                    route_pin=next(iter(route_pins)),
                    prior_calls=MappingProxyType(references),
                    result_output_ids=MappingProxyType(candidate_outputs),
                )
            finally:
                for lock in reversed(locks):
                    lock.release()

    def mark_arguments_complete(self, key: LedgerKey) -> LedgerSnapshot:
        """Record the complete provider call without storing its arguments."""

        return self._transition(key, CallState.STARTED, CallState.ARGUMENTS_COMPLETE)

    def mark_waiting_for_result(self, key: LedgerKey) -> LedgerSnapshot:
        """Make the complete call available to its harness."""

        return self._transition(
            key,
            CallState.ARGUMENTS_COMPLETE,
            CallState.WAITING_FOR_RESULT,
        )

    def accept_result(self, key: LedgerKey, identity: ResultIdentity) -> ResultReceipt:
        """Accept one digest or return the status of an exact retry."""

        if not isinstance(identity, ResultIdentity):
            raise TypeError("identity must be a ResultIdentity")
        with self._locked_entry(key) as entry:
            self._check_result_identity(entry, identity)
            if entry.result_digest is not None:
                if entry.result_digest != identity.result_digest:
                    raise ResultConflictError
                if (
                    entry.result_identity is None
                    or identity.request_id != entry.result_identity.request_id
                ):
                    raise ResultConflictError
                return ResultReceipt(disposition="retry", snapshot=self._snapshot(entry))

            if entry.state is not CallState.WAITING_FOR_RESULT:
                raise InvalidCallStateError

            entry.result_digest = identity.result_digest
            entry.result_identity = identity
            entry.continuation_request_ids = (identity.request_id,)
            entry.state = CallState.RESULT_RECEIVED
            return ResultReceipt(disposition="accepted", snapshot=self._snapshot(entry))

    def acquire_continuation(self, key: LedgerKey, *, result_digest: str) -> ContinuationTicket:
        """Own, join, or replay a continuation that has one required result."""

        if not _DIGEST.fullmatch(result_digest):
            raise ValueError("The result digest must be a lowercase SHA-256 digest")
        scope = ContinuationScope(
            principal_id=key.principal_id,
            project_id=key.project_id,
            harness=key.harness,
        )
        return self.acquire_group_continuation(
            scope,
            (key.call_id,),
            result_digests={key.call_id: result_digest},
        )

    def acquire_group_continuation(
        self,
        scope: ContinuationScope,
        call_ids: Sequence[str],
        *,
        result_digests: Mapping[str, str],
    ) -> ContinuationTicket:
        """Start one provider continuation after all group results arrive."""

        probe = self.probe_calls(scope, call_ids)
        ordered_call_ids = tuple(call_ids)
        if set(result_digests) != set(ordered_call_ids):
            raise IncompleteCallGroupError
        for result_digest in result_digests.values():
            if not _DIGEST.fullmatch(result_digest):
                raise ValueError("The result digest must be a lowercase SHA-256 digest")

        with self._table_lock:
            keys = self._group_keys(scope, probe.conversation_id, ordered_call_ids)
            expected_call_ids = {key.call_id for key in keys}
            if expected_call_ids != set(ordered_call_ids):
                raise IncompleteCallGroupError
            locks = [self._locks[key] for key in keys]
            for lock in locks:
                lock.acquire()
            try:
                entries = [self._get_locked_entry(key) for key in keys]
                for entry in entries:
                    call_id = entry.registration.key.call_id
                    recorded = entry.result_digest
                    supplied = result_digests[call_id]
                    if recorded is None:
                        raise IncompleteCallGroupError
                    if recorded != supplied:
                        raise ResultConflictError

                continuation_request_ids = {
                    entry.continuation_request_ids[-1]
                    for entry in entries
                    if entry.continuation_request_ids
                }
                if len(continuation_request_ids) != 1 or any(
                    not entry.continuation_request_ids for entry in entries
                ):
                    raise ConversationMismatchError

                states = {entry.state for entry in entries}
                if states == {CallState.RESULT_RECEIVED}:
                    future: Future[ContinuationOutcome] = Future()
                    for entry in entries:
                        entry.flight = future
                        entry.state = CallState.CONTINUING
                    return ContinuationTicket("owner", future)
                if states == {CallState.CONTINUING}:
                    flights = {id(entry.flight): entry.flight for entry in entries}
                    if len(flights) != 1 or None in flights.values():
                        raise InvalidCallStateError
                    return ContinuationTicket("join", next(iter(flights.values())))  # type: ignore[arg-type]
                if states <= {CallState.RESOLVED, CallState.ABANDONED}:
                    outcomes = {entry.terminal_outcome for entry in entries}
                    if len(outcomes) != 1 or None in outcomes:
                        raise InvalidCallStateError
                    future = Future()
                    future.set_result(next(iter(outcomes)))  # type: ignore[arg-type]
                    return ContinuationTicket("replay", future)
                raise InvalidCallStateError
            finally:
                for lock in reversed(locks):
                    lock.release()

    def complete_continuation(
        self,
        key: LedgerKey,
        outcome: ContinuationOutcome,
    ) -> LedgerSnapshot:
        """Complete the provider continuation and retain a replay tombstone."""

        if not isinstance(outcome, ContinuationOutcome):
            raise TypeError("outcome must be a ContinuationOutcome")
        with self._table_lock:
            target = self._entries.get(key)
            if target is None:
                raise UnknownCallError
            flight = target.flight
            if target.state is not CallState.CONTINUING or flight is None:
                raise InvalidCallStateError
            entries = [entry for entry in self._entries.values() if entry.flight is flight]
            locks = [self._locks[entry.registration.key] for entry in entries]
            for lock in locks:
                lock.acquire()
            try:
                expires_at = self._now() + self._tombstone_ttl
                for entry in entries:
                    if entry.state is not CallState.CONTINUING or entry.flight is not flight:
                        raise InvalidCallStateError
                    entry.state = outcome.state
                    entry.terminal_outcome = outcome
                    entry.tombstone = True
                    entry.expires_at = expires_at
                    entry.flight = None
                flight.set_result(outcome)
                return self._snapshot(target)
            finally:
                for lock in reversed(locks):
                    lock.release()

    def abandon(self, key: LedgerKey, *, error_code: str) -> LedgerSnapshot:
        """Abandon an incomplete call and retain a safe tombstone."""

        outcome = ContinuationOutcome(state=CallState.ABANDONED, error_code=error_code)
        with self._table_lock:
            target = self._entries.get(key)
            if target is not None and target.flight is not None:
                return self.complete_continuation(key, outcome)
        with self._locked_entry(key) as entry:
            if entry.state in {CallState.RESOLVED, CallState.ABANDONED}:
                if entry.terminal_outcome == outcome:
                    return self._snapshot(entry)
                raise InvalidCallStateError
            entry.state = CallState.ABANDONED
            entry.terminal_outcome = outcome
            entry.tombstone = True
            entry.expires_at = self._now() + self._tombstone_ttl
            if entry.flight is not None:
                entry.flight.set_result(outcome)
                entry.flight = None
            return self._snapshot(entry)

    def get(self, key: LedgerKey) -> LedgerSnapshot:
        """Get one immutable secret-free ledger snapshot."""

        with self._locked_entry(key) as entry:
            return self._snapshot(entry)

    def cleanup(self) -> int:
        """Remove tombstones after their retention period."""

        now = self._now()
        removed = 0
        with self._table_lock:
            for key, entry in tuple(self._entries.items()):
                lock = self._locks[key]
                with lock:
                    if not entry.tombstone and now >= entry.expires_at:
                        self._expire(entry, now)
                        continue
                    if entry.tombstone and now >= entry.expires_at:
                        self._remove(key, entry)
                        removed += 1
        return removed

    @property
    def size(self) -> int:
        """Return the current number of active entries and tombstones."""

        with self._table_lock:
            return len(self._entries)

    @property
    def states(self) -> Mapping[LedgerKey, CallState]:
        """Return a read-only copy of the current call states."""

        with self._table_lock:
            return MappingProxyType({key: entry.state for key, entry in self._entries.items()})

    def _transition(
        self,
        key: LedgerKey,
        expected: CallState,
        target: CallState,
    ) -> LedgerSnapshot:
        with self._locked_entry(key) as entry:
            if entry.state is target:
                return self._snapshot(entry)
            if entry.state is not expected:
                raise InvalidCallStateError
            entry.state = target
            return self._snapshot(entry)

    def _locked_entry(self, key: LedgerKey) -> _EntryContext:
        if not isinstance(key, LedgerKey):
            raise TypeError("key must be a LedgerKey")
        return _EntryContext(self, key)

    def _get_locked_entry(self, key: LedgerKey) -> _Entry:
        entry = self._entries.get(key)
        if entry is None:
            raise UnknownCallError
        if self._is_expired_tombstone(entry):
            raise ContinuationExpiredError
        now = self._now()
        if now >= entry.expires_at:
            if entry.tombstone:
                self._remove(key, entry)
            else:
                self._expire(entry, now)
            raise ContinuationExpiredError
        return entry

    @staticmethod
    def _is_expired_tombstone(entry: _Entry) -> bool:
        outcome = entry.terminal_outcome
        return (
            entry.tombstone
            and outcome is not None
            and outcome.error_code == "tool_continuation_expired"
        )

    def _expire(self, entry: _Entry, now: float) -> None:
        outcome = ContinuationOutcome(
            state=CallState.ABANDONED,
            error_code="tool_continuation_expired",
        )
        entry.state = CallState.ABANDONED
        entry.terminal_outcome = outcome
        entry.tombstone = True
        entry.expires_at = now + self._tombstone_ttl
        if entry.flight is not None:
            if not entry.flight.done():
                entry.flight.set_result(outcome)
            entry.flight = None

    def _remove(self, key: LedgerKey, entry: _Entry) -> None:
        self._entries.pop(key, None)
        self._locks.pop(key, None)
        self._index.pop(self._index_key(entry.registration), None)
        source_key = self._source_key(entry.registration)
        if source_key is not None:
            self._source_index.pop(source_key, None)
        scope_key = self._scope_key(key)
        scoped = self._scope_index.get(scope_key)
        if scoped is not None:
            scoped.discard(key)
            if not scoped:
                self._scope_index.pop(scope_key, None)

    @staticmethod
    def _check_result_identity(entry: _Entry, identity: ResultIdentity) -> None:
        registration = entry.registration
        accepted_output_id = (
            entry.result_identity.source_output_id
            if entry.result_identity is not None
            else None
        )
        parent_request_id = (
            registration.prior_continuation_request_ids[-1]
            if registration.prior_continuation_request_ids
            else registration.initial_request_id
        )
        if (
            identity.parent_request_id != parent_request_id
            or identity.choice_index != registration.choice_index
            or identity.tool_call_index != registration.tool_call_index
            or (
                accepted_output_id is not None
                and identity.source_output_id != accepted_output_id
            )
        ):
            raise ConversationMismatchError

    @staticmethod
    def _index_key(registration: CallRegistration) -> tuple[str, str, str, str, int, str, int]:
        key = registration.key
        return (
            key.principal_id,
            key.project_id,
            key.harness,
            key.conversation_id,
            registration.choice_index,
            registration.output_id,
            registration.tool_call_index,
        )

    @staticmethod
    def _scope_key(key: LedgerKey) -> tuple[str, str, str, str]:
        return (key.principal_id, key.project_id, key.harness, key.call_id)

    @staticmethod
    def _source_key(
        registration: CallRegistration,
    ) -> tuple[str, str, str, str, str, str] | None:
        source_call_id = registration.source_call_id
        if source_call_id is None:
            return None
        key = registration.key
        return (
            key.principal_id,
            key.project_id,
            key.harness,
            key.conversation_id,
            registration.output_id,
            source_call_id,
        )

    def _group_keys(
        self,
        scope: ContinuationScope,
        conversation_id: str,
        call_ids: tuple[str, ...],
    ) -> list[LedgerKey]:
        requested = [
            self._entries[
                next(
                    iter(
                        self._scope_index[
                            (scope.principal_id, scope.project_id, scope.harness, call_id)
                        ]
                    )
                )
            ]
            for call_id in call_ids
        ]
        group_keys = {self._continuation_group_key(entry.registration) for entry in requested}
        if len(group_keys) != 1:
            raise ConversationMismatchError
        group_key = next(iter(group_keys))
        return [
            key
            for key, entry in self._entries.items()
            if key.principal_id == scope.principal_id
            and key.project_id == scope.project_id
            and key.harness == scope.harness
            and key.conversation_id == conversation_id
            and self._continuation_group_key(entry.registration) == group_key
        ]

    @classmethod
    def _continuation_group_key(
        cls,
        registration: CallRegistration,
    ) -> tuple[str, int, str, str | None]:
        return (
            cls._call_request_id(registration),
            registration.choice_index,
            registration.output_id,
            registration.parallel_group_id,
        )

    @staticmethod
    def _call_request_id(registration: CallRegistration) -> str:
        if registration.prior_continuation_request_ids:
            return registration.prior_continuation_request_ids[-1]
        return registration.initial_request_id

    @staticmethod
    def _snapshot(entry: _Entry) -> LedgerSnapshot:
        registration = entry.registration
        return LedgerSnapshot(
            key=registration.key,
            source_call_id=registration.source_call_id,
            state=entry.state,
            result_digest=entry.result_digest,
            route_pin=registration.route_pin,
            expires_at=entry.expires_at,
            initial_request_id=registration.initial_request_id,
            continuation_request_ids=(
                *registration.prior_continuation_request_ids,
                *entry.continuation_request_ids,
            ),
            choice_index=registration.choice_index,
            output_id=registration.output_id,
            source_result_output_id=(
                entry.result_identity.source_output_id if entry.result_identity is not None else None
            ),
            tool_call_index=registration.tool_call_index,
            parallel_group_id=registration.parallel_group_id,
            tombstone=entry.tombstone,
            terminal_outcome=entry.terminal_outcome,
        )

    def _now(self) -> float:
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)):
            raise RuntimeError("The ledger clock returned an invalid time")
        numeric_now = float(now)
        if not math.isfinite(numeric_now):
            raise RuntimeError("The ledger clock returned an invalid time")
        return numeric_now


class _EntryContext:
    """Acquire the table lock and then the stable per-call lock."""

    __slots__ = ("_entry", "_key", "_ledger", "_lock")

    def __init__(self, ledger: ContinuationLedger, key: LedgerKey) -> None:
        self._ledger = ledger
        self._key = key
        self._lock: threading.RLock | None = None
        self._entry: _Entry | None = None

    def __enter__(self) -> _Entry:
        with self._ledger._table_lock:
            lock = self._ledger._locks.get(self._key)
            if lock is None:
                raise UnknownCallError
            lock.acquire()
            self._lock = lock
            try:
                self._entry = self._ledger._get_locked_entry(self._key)
            except BaseException:
                lock.release()
                self._lock = None
                raise
            return self._entry

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._lock is not None:
            self._lock.release()


def _require_identifier(value: object) -> None:
    if (
        type(value) is not str
        or not 1 <= len(value) <= 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError("A ledger identifier is invalid")


def _require_index(value: object) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("A ledger index must be a nonnegative integer")


def _require_ttl(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("A ledger retention period must be a positive number")
    numeric_value = float(value)
    if not math.isfinite(numeric_value) or numeric_value <= 0:
        raise ValueError("A ledger retention period must be a positive number")
    return numeric_value

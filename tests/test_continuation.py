from __future__ import annotations

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

import pytest

from ppmlx.continuation import (
    CallConflictError,
    CallRegistration,
    CallState,
    ContinuationExpiredError,
    ContinuationLedger,
    ContinuationOutcome,
    ContinuationScope,
    ConversationMismatchError,
    IncompleteCallGroupError,
    InvalidCallStateError,
    LedgerCapacityError,
    LedgerKey,
    ResultConflictError,
    ResultIdentity,
    RoutePin,
    UnknownCallError,
)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def digest(value: bytes = b"tool result") -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def registration(**changes: Any) -> CallRegistration:
    value = CallRegistration(
        key=LedgerKey(
            principal_id="principal-a",
            project_id="project-a",
            harness="codex-0.147.0",
            conversation_id="conversation-a",
            call_id="call-a",
        ),
        source_call_id="source-call-a",
        tool_name="read_file",
        initial_request_id="request-a",
        choice_index=0,
        output_id="output-a",
        tool_call_index=0,
        parallel_group_id="parallel-a",
        route_pin=RoutePin(
            decision_id="route-a",
            provider="local",
            model="qwen3-30b",
            candidate_id="candidate-a",
        ),
    )
    return replace(value, **changes)


def result_identity(**changes: Any) -> ResultIdentity:
    value = ResultIdentity(
        request_id="request-b",
        parent_request_id="request-a",
        choice_index=0,
        tool_call_index=0,
        result_digest=digest(),
        source_output_id="result-output-a",
    )
    return replace(value, **changes)


def ready_ledger() -> tuple[ContinuationLedger, CallRegistration]:
    ledger = ContinuationLedger()
    call = registration()
    ledger.register_call(call)
    ledger.mark_arguments_complete(call.key)
    ledger.mark_waiting_for_result(call.key)
    return ledger, call


def scope() -> ContinuationScope:
    return ContinuationScope(
        principal_id="principal-a",
        project_id="project-a",
        harness="codex-0.147.0",
    )


def test_call_lifecycle_preserves_stable_identity_and_route() -> None:
    ledger, call = ready_ledger()

    receipt = ledger.accept_result(call.key, result_identity())
    ticket = ledger.acquire_continuation(call.key, result_digest=digest())
    terminal = ledger.complete_continuation(
        call.key,
        ContinuationOutcome(state=CallState.RESOLVED),
    )

    assert receipt.disposition == "accepted"
    assert ticket.disposition == "owner"
    assert ticket.result(timeout=0) == ContinuationOutcome(state=CallState.RESOLVED)
    assert terminal == ledger.get(call.key)
    assert terminal.state is CallState.RESOLVED
    assert terminal.tombstone is True
    assert terminal.key == call.key
    assert terminal.route_pin == call.route_pin
    assert terminal.initial_request_id == "request-a"
    assert terminal.continuation_request_ids == ("request-b",)
    assert terminal.choice_index == 0
    assert terminal.output_id == "output-a"
    assert terminal.source_result_output_id == "result-output-a"
    assert terminal.tool_call_index == 0
    assert terminal.parallel_group_id == "parallel-a"


def test_probe_resolves_adapter_context_without_a_conversation_id() -> None:
    ledger, call = ready_ledger()

    probe = ledger.probe_calls(
        scope(),
        (call.key.call_id,),
        result_output_ids={call.key.call_id: "result-output-a"},
    )

    assert probe.conversation_id == call.key.conversation_id
    assert probe.parent_request_id == call.initial_request_id
    assert probe.route_pin == call.route_pin
    assert probe.prior_calls[call.key.call_id].call_id == call.key.call_id
    assert probe.prior_calls[call.key.call_id].name == call.tool_name
    assert probe.prior_calls[call.key.call_id].output_id == call.output_id
    assert probe.result_output_ids == {call.key.call_id: "result-output-a"}
    with pytest.raises(TypeError):
        probe.result_output_ids[call.key.call_id] = "changed"  # type: ignore[index]


def test_probe_scope_does_not_search_another_project() -> None:
    ledger, call = ready_ledger()

    with pytest.raises(UnknownCallError, match="tool_call_unknown"):
        ledger.probe_calls(replace(scope(), project_id="project-b"), (call.key.call_id,))


def test_state_transitions_are_ordered_and_idempotent() -> None:
    ledger = ContinuationLedger()
    call = registration()

    assert ledger.register_call(call).state is CallState.STARTED
    assert ledger.register_call(call).state is CallState.STARTED
    assert ledger.mark_arguments_complete(call.key).state is CallState.ARGUMENTS_COMPLETE
    assert ledger.mark_arguments_complete(call.key).state is CallState.ARGUMENTS_COMPLETE
    assert ledger.mark_waiting_for_result(call.key).state is CallState.WAITING_FOR_RESULT
    assert ledger.mark_waiting_for_result(call.key).state is CallState.WAITING_FOR_RESULT

    with pytest.raises(InvalidCallStateError, match="tool_call_invalid_state"):
        ledger.mark_arguments_complete(call.key)


def test_exact_concurrent_retry_joins_one_continuation() -> None:
    ledger, call = ready_ledger()
    ledger.accept_result(call.key, result_identity())

    with ThreadPoolExecutor(max_workers=12) as pool:
        tickets = list(
            pool.map(
                lambda _: ledger.acquire_continuation(call.key, result_digest=digest()),
                range(24),
            )
        )

    assert sum(ticket.disposition == "owner" for ticket in tickets) == 1
    assert sum(ticket.disposition == "join" for ticket in tickets) == 23

    outcome = ContinuationOutcome(state=CallState.RESOLVED)
    ledger.complete_continuation(call.key, outcome)
    assert [ticket.result(timeout=0) for ticket in tickets] == [outcome] * 24


@pytest.mark.asyncio
async def test_cancelled_join_wait_does_not_cancel_the_shared_flight() -> None:
    ledger, call = ready_ledger()
    ledger.accept_result(call.key, result_identity())
    owner = ledger.acquire_continuation(call.key, result_digest=digest())
    joined = ledger.acquire_continuation(call.key, result_digest=digest())
    wait_task = asyncio.create_task(joined.wait())
    await asyncio.sleep(0)

    wait_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await wait_task

    assert owner.done is False
    outcome = ContinuationOutcome(state=CallState.RESOLVED)
    ledger.complete_continuation(call.key, outcome)
    assert await owner.wait(timeout=0.1) == outcome


def test_parallel_results_start_one_shared_continuation() -> None:
    ledger, first = ready_ledger()
    second = registration(
        key=replace(first.key, call_id="call-b"),
        source_call_id="source-call-b",
        tool_name="write_file",
        tool_call_index=1,
    )
    ledger.register_call(second)
    ledger.mark_arguments_complete(second.key)
    ledger.mark_waiting_for_result(second.key)
    ledger.probe_calls(
        scope(),
        (first.key.call_id, second.key.call_id),
        result_output_ids={
            first.key.call_id: "result-output-a",
            second.key.call_id: "result-output-b",
        },
    )
    first_result = result_identity()
    second_result = result_identity(
        request_id="request-b",
        tool_call_index=1,
        result_digest=digest(b"second"),
        source_output_id="result-output-b",
    )
    ledger.accept_result(first.key, first_result)

    with pytest.raises(IncompleteCallGroupError, match="tool_results_incomplete"):
        ledger.acquire_group_continuation(
            scope(),
            (first.key.call_id,),
            result_digests={first.key.call_id: first_result.result_digest},
        )

    ledger.accept_result(second.key, second_result)
    call_ids = (first.key.call_id, second.key.call_id)
    result_digests = {
        first.key.call_id: first_result.result_digest,
        second.key.call_id: second_result.result_digest,
    }
    owner = ledger.acquire_group_continuation(
        scope(), call_ids, result_digests=result_digests
    )
    joined = ledger.acquire_group_continuation(
        scope(), call_ids, result_digests=result_digests
    )

    assert owner.disposition == "owner"
    assert joined.disposition == "join"
    outcome = ContinuationOutcome(state=CallState.RESOLVED)
    ledger.complete_continuation(first.key, outcome)
    assert owner.result(timeout=0) == outcome
    assert joined.result(timeout=0) == outcome
    assert ledger.get(first.key).state is CallState.RESOLVED
    assert ledger.get(second.key).state is CallState.RESOLVED


def test_parallel_group_rejects_a_conflicting_digest() -> None:
    ledger, call = ready_ledger()
    ledger.accept_result(call.key, result_identity())

    with pytest.raises(ResultConflictError, match="tool_result_conflict"):
        ledger.acquire_group_continuation(
            scope(),
            (call.key.call_id,),
            result_digests={call.key.call_id: digest(b"conflict")},
        )


def test_later_exact_retry_uses_recorded_status() -> None:
    ledger, call = ready_ledger()
    identity = result_identity()
    ledger.accept_result(call.key, identity)
    ledger.acquire_continuation(call.key, result_digest=identity.result_digest)
    outcome = ContinuationOutcome(state=CallState.RESOLVED)
    ledger.complete_continuation(call.key, outcome)

    receipt = ledger.accept_result(call.key, identity)
    ticket = ledger.acquire_continuation(call.key, result_digest=identity.result_digest)

    assert receipt.disposition == "retry"
    assert ticket.disposition == "replay"
    assert ticket.result(timeout=0) == outcome


def test_different_result_digest_fails_without_a_second_continuation() -> None:
    ledger, call = ready_ledger()
    ledger.accept_result(call.key, result_identity())
    owner = ledger.acquire_continuation(call.key, result_digest=digest())

    with pytest.raises(ResultConflictError, match="tool_result_conflict"):
        ledger.accept_result(call.key, result_identity(result_digest=digest(b"different")))
    with pytest.raises(ResultConflictError, match="tool_result_conflict"):
        ledger.acquire_continuation(call.key, result_digest=digest(b"different"))

    assert owner.disposition == "owner"
    assert owner.done is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("parent_request_id", "other-request"),
        ("choice_index", 1),
        ("tool_call_index", 1),
    ],
)
def test_result_identity_mismatch_fails(field: str, value: object) -> None:
    ledger, call = ready_ledger()

    with pytest.raises(ConversationMismatchError, match="tool_conversation_mismatch"):
        ledger.accept_result(call.key, result_identity(**{field: value}))


def test_result_can_link_to_an_existing_continuation_request() -> None:
    call = registration(prior_continuation_request_ids=("request-before-call",))
    ledger = ContinuationLedger()
    ledger.register_call(call)
    ledger.mark_arguments_complete(call.key)
    ledger.mark_waiting_for_result(call.key)

    receipt = ledger.accept_result(
        call.key,
        result_identity(parent_request_id="request-before-call"),
    )

    assert receipt.snapshot.initial_request_id == "request-a"
    assert receipt.snapshot.continuation_request_ids == (
        "request-before-call",
        "request-b",
    )


def test_later_retry_can_use_a_new_linked_request_id() -> None:
    ledger, call = ready_ledger()
    ledger.accept_result(call.key, result_identity())

    retry = ledger.accept_result(call.key, result_identity(request_id="request-c"))

    assert retry.disposition == "retry"
    assert retry.snapshot.continuation_request_ids == ("request-b", "request-c")


def test_retry_must_keep_the_result_output_id() -> None:
    ledger, call = ready_ledger()
    ledger.probe_calls(
        scope(),
        (call.key.call_id,),
        result_output_ids={call.key.call_id: "result-output-a"},
    )
    ledger.accept_result(call.key, result_identity())

    with pytest.raises(ConversationMismatchError, match="tool_conversation_mismatch"):
        ledger.accept_result(call.key, result_identity(source_output_id="result-output-b"))


@pytest.mark.parametrize(
    "key_change",
    [
        {"principal_id": "principal-b"},
        {"project_id": "project-b"},
        {"harness": "claude-code-2.1.231"},
        {"conversation_id": "conversation-b"},
    ],
)
def test_call_lookup_isolated_by_all_scope_fields(key_change: dict[str, str]) -> None:
    ledger, call = ready_ledger()
    other_key = replace(call.key, **key_change)

    with pytest.raises(UnknownCallError, match="tool_call_unknown"):
        ledger.get(other_key)


def test_parallel_calls_require_distinct_tool_indexes() -> None:
    ledger = ContinuationLedger()
    first = registration()
    ledger.register_call(first)
    other_key = replace(first.key, call_id="call-b")

    with pytest.raises(CallConflictError, match="tool_call_conflict"):
        ledger.register_call(registration(key=other_key, source_call_id="source-call-b"))

    second = registration(
        key=other_key,
        source_call_id="source-call-b",
        tool_call_index=1,
    )
    assert ledger.register_call(second).tool_call_index == 1


def test_tool_index_can_repeat_for_a_different_output() -> None:
    ledger = ContinuationLedger()
    first = registration()
    ledger.register_call(first)
    second = registration(
        key=replace(first.key, call_id="call-b"),
        source_call_id="source-call-b",
        output_id="output-b",
    )

    assert ledger.register_call(second).tool_call_index == first.tool_call_index


def test_same_call_id_cannot_change_stable_registration_data() -> None:
    ledger = ContinuationLedger()
    call = registration()
    ledger.register_call(call)

    with pytest.raises(CallConflictError, match="tool_call_conflict"):
        ledger.register_call(replace(call, output_id="other-output"))


def test_source_call_id_is_unique_in_one_conversation() -> None:
    ledger = ContinuationLedger()
    first = registration()
    ledger.register_call(first)
    second = registration(
        key=replace(first.key, call_id="call-b"),
        tool_call_index=1,
    )

    with pytest.raises(CallConflictError, match="tool_call_conflict"):
        ledger.register_call(second)


def test_active_expiry_creates_a_minimal_tombstone_then_cleanup_removes_it() -> None:
    clock = Clock()
    ledger = ContinuationLedger(active_ttl_seconds=5, clock=clock)
    call = registration()
    ledger.register_call(call)
    clock.now += 5

    with pytest.raises(ContinuationExpiredError, match="tool_continuation_expired"):
        ledger.get(call.key)

    assert ledger.states[call.key] is CallState.ABANDONED
    assert ledger.size == 1
    clock.now += 1
    with pytest.raises(ContinuationExpiredError, match="tool_continuation_expired"):
        ledger.get(call.key)
    clock.now += 86_398
    assert ledger.cleanup() == 0
    clock.now += 1
    assert ledger.cleanup() == 1
    assert ledger.size == 0

    with pytest.raises(UnknownCallError, match="tool_call_unknown"):
        ledger.get(call.key)


def test_terminal_tombstone_is_retained_for_its_own_period() -> None:
    clock = Clock()
    ledger = ContinuationLedger(active_ttl_seconds=100, clock=clock)
    call = registration()
    ledger.register_call(call)
    terminal = ledger.abandon(call.key, error_code="incomplete_tool_call")

    assert terminal.state is CallState.ABANDONED
    assert terminal.tombstone is True
    assert terminal.expires_at == 87_400.0
    clock.now = 87_399.0
    assert ledger.cleanup() == 0
    clock.now = 87_400.0
    assert ledger.cleanup() == 1


def test_ledger_applies_backpressure_without_evicting_valid_tombstones() -> None:
    clock = Clock()
    ledger = ContinuationLedger(max_entries=2, clock=clock)
    first = registration()
    second = registration(
        key=replace(first.key, call_id="call-b"),
        source_call_id="source-call-b",
        output_id="output-b",
        tool_call_index=1,
    )
    ledger.register_calls((first, second))
    ledger.abandon(first.key, error_code="test_terminal")
    ledger.abandon(second.key, error_code="test_terminal")
    third = registration(
        key=replace(first.key, call_id="call-c"),
        source_call_id="source-call-c",
        output_id="output-c",
        tool_call_index=2,
    )

    with pytest.raises(LedgerCapacityError, match="continuation_capacity_exceeded"):
        ledger.register_call(third)

    assert ledger.size == 2
    assert ledger.get(first.key).tombstone is True
    clock.now += 86_400
    assert ledger.cleanup() == 2
    assert ledger.register_call(third).state is CallState.STARTED


def test_group_registration_rolls_back_when_capacity_is_unavailable() -> None:
    ledger = ContinuationLedger(max_entries=1)
    first = registration()
    second = registration(
        key=replace(first.key, call_id="call-b"),
        source_call_id="source-call-b",
        output_id="output-b",
        tool_call_index=1,
    )

    with pytest.raises(LedgerCapacityError, match="continuation_capacity_exceeded"):
        ledger.register_calls((first, second))

    assert ledger.size == 0


def test_abandon_wakes_all_joined_retries() -> None:
    ledger, call = ready_ledger()
    ledger.accept_result(call.key, result_identity())
    owner = ledger.acquire_continuation(call.key, result_digest=digest())
    joined = ledger.acquire_continuation(call.key, result_digest=digest())

    terminal = ledger.abandon(call.key, error_code="provider_timeout")

    expected = ContinuationOutcome(state=CallState.ABANDONED, error_code="provider_timeout")
    assert terminal.terminal_outcome == expected
    assert owner.result(timeout=0) == expected
    assert joined.result(timeout=0) == expected


def test_ledger_does_not_accept_or_expose_raw_arguments_or_results() -> None:
    ledger, call = ready_ledger()
    secret = "sk-secret-value"
    identity = result_identity(result_digest=digest(secret.encode()))
    snapshot = ledger.accept_result(call.key, identity).snapshot

    assert secret not in repr(ledger)
    assert secret not in repr(snapshot)
    assert not hasattr(snapshot, "arguments")
    assert not hasattr(snapshot, "result")
    assert snapshot.result_digest == digest(secret.encode())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: LedgerKey("bad\nvalue", "p", "h", "c", "x"),
        lambda: RoutePin("route", "provider", "model", "token\nsecret"),
        lambda: ResultIdentity("r", "p", 0, 0, "not-a-digest"),
        lambda: ContinuationOutcome(CallState.RESOLVED, "unexpected_error"),
        lambda: ContinuationOutcome(CallState.ABANDONED, "unsafe error text"),
    ],
)
def test_public_records_reject_unsafe_or_invalid_values(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_public_errors_do_not_contain_identifiers_or_secret_data() -> None:
    ledger, call = ready_ledger()
    secret_digest = digest(b"secret result")
    ledger.accept_result(call.key, result_identity(result_digest=secret_digest))

    with pytest.raises(ResultConflictError) as captured:
        ledger.accept_result(call.key, result_identity(result_digest=digest(b"other secret")))

    message = str(captured.value)
    assert message == "Continuation ledger error tool_result_conflict"
    assert call.key.call_id not in message
    assert secret_digest not in message


def test_states_returns_a_read_only_copy() -> None:
    ledger = ContinuationLedger()
    call = registration()
    ledger.register_call(call)
    states = ledger.states

    with pytest.raises(TypeError):
        states[call.key] = CallState.RESOLVED  # type: ignore[index]
    ledger.mark_arguments_complete(call.key)
    assert states[call.key] is CallState.STARTED


def test_registration_rolls_back_after_duplicate_parallel_index() -> None:
    ledger = ContinuationLedger()
    first = registration()
    second_key = replace(first.key, call_id="call-b")
    ledger.register_call(first)

    with pytest.raises(CallConflictError):
        ledger.register_call(registration(key=second_key, source_call_id="source-call-b"))

    with pytest.raises(UnknownCallError):
        ledger.get(second_key)
    assert ledger.size == 1

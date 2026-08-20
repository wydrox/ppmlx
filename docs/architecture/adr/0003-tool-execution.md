# ADR 0003: Tool Execution

- Status: Accepted; amended by [ADR 0009](0009-bounded-tool-argument-repair.md)
- Date: 2026-08-18

## Context

Agent harnesses already control files, shells, network access, sandboxes, and user approval.
Providers emit tool calls in different formats and stream their arguments in different event types.

If ppmlx executes harness tools, it bypasses the harness security model.
If ppmlx changes call identifiers or accepted arguments after model-profile normalization, the harness cannot return a valid result.

## Decision

The harness is the tool owner and the only tool executor in the Phase 1 contract.
ppmlx transports tool definitions, tool calls, and tool results through the Agent IR.
It does not run a harness tool.

ADR 0009 permits one bounded, profile-declared syntax repair before Agent IR accepts a local model tool call.
That repair does not change tool ownership, call identity, result linking, or harness approval.

One tool round trip has four ordered states:

- The harness sends tool definitions with the request.
- The provider returns a tool call through ppmlx.
- The harness runs the tool and returns a result with the same `call_id`.
- The provider returns a final answer through ppmlx.

After the provider starts a tool call, its `call_id` is stable in all related stream events and later states.
If a source protocol has no call identifier, ppmlx creates one and stores the source mapping in the continuation ledger.

## Conversation and continuation ledger

The Agent IR uses an opaque `conversation_id` for state that spans HTTP requests.
The ingress adapter derives it from a native conversation or continuation identifier when the protocol supplies one.
Otherwise, ppmlx creates a random identifier before the first tool ledger entry.
The adapter preserves the native identifier in its private mapping or a namespaced extension.

The ledger key contains the local principal, harness, `conversation_id`, and `call_id`.
The local principal comes from local endpoint authentication in ADR 0004.
A localhost request without authentication uses one listener-scoped principal and has the limits in ADR 0004.

The ledger records the source identifier, state, result digest, route, and expiry time.
It also records the initial request, linked continuation requests, choice, output, tool index, and optional parallel group.
It does not need raw arguments or raw results after the active continuation ends.
Source identifiers that differ from `call_id` remain in a private adapter mapping or a namespaced extension.

The ledger uses these call states:

- `started`: The provider started the tool call.
- `arguments_complete`: ppmlx received the complete call.
- `waiting_for_result`: The harness can run the tool.
- `result_received`: ppmlx accepted one result.
- `continuing`: One provider continuation is active.

It uses these terminal states:

- `resolved`: The provider returned a terminal answer.
- `abandoned`: The call cannot continue.

## Normative rules

### Ownership and definitions

- ppmlx MUST NOT execute a tool that a harness supplies.
- ppmlx MUST NOT interpret a tool schema as permission to run that tool.
- ppmlx MUST preserve the tool name, description, input schema, order, and selection rule.
- ppmlx MUST preserve `arguments_raw` after the model profile accepts it under ADR 0002 and ADR 0009.
- ppmlx MUST emit a tool call only after its name and stable `call_id` are known.
- Each accepted argument delta MUST keep source order.

### Call state

- A `tool_result` MUST refer to an existing unresolved `call_id` in the same conversation.
- ppmlx MUST reject an unknown `call_id` or a call from a different conversation.
- ppmlx MUST preserve multiple tool-call order and separate each argument stream.
- ppmlx MUST preserve an explicit tool error as `is_error=true`.
- ppmlx MUST NOT convert a tool error to assistant text.
- An adapter MUST keep the provider stop reason that requests tool use.

### Conversation identity

- The normalizer MUST add `conversation_id` before it creates a ledger entry.
- A continuation MUST use the same local principal, harness, and conversation.
- A continuation MUST link its `parent_request_id` to an earlier request in the same conversation.
- A tool result event MUST use the continuation `request_id`.
- ppmlx MUST NOT match a result to a call from another conversation.

Call identity rules also apply:

- The tool-call lifecycle MUST match the original `choice_index`, `output_id`, `tool_call_index`, and `call_id`.
- A tool result MUST match the original `choice_index`, `tool_call_index`, and `call_id`.
- The ledger records a separate source `output_id` for a tool result when the source supplies one.
- A missing or different identity field MUST cause `tool_conversation_mismatch`.

### Concurrency and duplicates

- Parallel calls MUST use different `call_id` values in one conversation.
- Parallel calls MUST use different `tool_call_index` values for one choice and output.
- One parallel call group MUST keep the same `parallel_group_id` in all related events.
- ppmlx MUST lock each call transition and each provider continuation.
- The first valid result changes the call to `result_received`.

Retry rules also apply:

- An exact concurrent retry MUST join the existing single-flight continuation.
- A later result with the same content digest MUST return the recorded status without another provider request.
- A result with a different digest MUST cause `tool_result_conflict`.

### Lifetime and cleanup

- An unresolved call MUST expire after a configurable period with a default value of 24 hours.
- A resolved or abandoned call MUST keep a replay tombstone for 24 hours.
- An expired call MUST cause `tool_continuation_expired`.
- Cleanup MUST remove raw arguments and results before it removes the minimal tombstone.
- Cleanup MUST follow the retention and redaction rules in ADR 0007.
- A process restart MAY expire an in-memory ledger, but it MUST return a clear expiry error.

### Representation and capabilities

- A required tool call MUST NOT become a plain-text suggestion.
- Plain text that resembles a tool call MUST NOT become a tool call without a documented parser contract.
- A model profile that permits argument repair MUST follow ADR 0009.
- A provider adapter MUST declare support for tools, parallel calls, strict schemas, and streamed arguments.
- Routing MUST reject a tool request when the selected route lacks a required tool capability.

### Completion and failure

- ppmlx MUST NOT send the next model turn before it receives all required tool results.
- A disconnect before `tool_call.completed` MUST mark the incomplete call as `abandoned`.
- A disconnect after `tool_call.completed` MUST keep the call until its result deadline.
- A disconnect after result acceptance MUST NOT start a second provider continuation.
- A retry MUST join the active continuation or use its retained terminal outcome.
- A timeout MUST leave a typed incomplete-call error in sanitized diagnostics.

## Security and privacy

The system treats tool output as untrusted model input.
ppmlx must preserve its role as a tool result and must not promote it to a system instruction.

The harness remains responsible for sandbox controls, path controls, network controls, and user approval.
ppmlx must not claim that a provider request grants local tool permission.
A bounded syntax repair does not increase trust or grant tool permission.

Tool arguments and results can contain secrets.
Diagnostics must redact them by default and use only request and call identifiers.
The router must apply data policy before it sends a tool result to a remote provider.

## Consequences

Existing harness approval and sandbox behavior stays active.
ppmlx avoids a second tool runtime and a second permission model.

The proxy needs a conversation and continuation ledger that spans multiple HTTP requests.
Adapters need strict stream state machines and explicit capability data.
The system cannot hide provider defects with a lossy plain-text fallback.
Any bounded repair must be explicit, deterministic, profile-specific, and auditable under ADR 0009.

## Compatibility effects

- **Claude Code:** `tool_use.id` maps to `call_id`, and the matching `tool_result.tool_use_id` uses the same value.
- **Codex:** Responses function-call items and `function_call_output` items use one stable `call_id`.
- **OpenCode:** Assistant `tool_calls[].id` maps to `call_id`. A tool message returns the matching identifier.
- **Pi:** Assistant `tool_calls[].id` maps to `call_id`. Pi runs the tool and returns the matching tool message.

Each harness keeps its native approval user interface.
An adapter can change event shape, but it cannot change call identity, order, accepted arguments, or result status.

## Rejected alternatives

### ppmlx tool runtime

This option executes tools inside the proxy.
It duplicates harness sandboxes and can bypass user approval.

### Text tool-call fallback

This option asks a harness to parse a tool call from assistant text.
It loses structured identity and creates unsafe ambiguity.

### New identifier at each adapter

This option creates a different call identifier at each protocol boundary.
It makes result matching and trace diagnosis unreliable.

### Unbounded or hidden automatic argument repair

This option changes malformed arguments without an explicit profile, one-repair budget, or audit record.
It hides provider defects and breaks reproducible transport.
ADR 0009 permits only one bounded syntax repair before Agent IR acceptance.

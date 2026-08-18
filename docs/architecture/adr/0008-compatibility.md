# ADR 0008: Compatibility

- Status: Accepted
- Date: 2026-08-18
- Phase: 1, contract freeze

## Context

Claude Code, Codex, OpenCode, and Pi use different request, stream, and tool formats. Provider models can also emit different tool-call forms.

A shape-only conversion is not sufficient. ppmlx must preserve the meaning, identity, order, error state, and stream state of every tool action.

Harnesses and provider APIs change. Compatibility claims need exact versions and replayable evidence.

## Decision

ppmlx uses versioned contract fixtures for each harness and protocol. The fixtures define the supported behavior at a tested version boundary.

Each protocol adapter maps native data to `agent-ir/v1` from ADR 0002. It then maps Agent IR output to the native harness protocol.

Tool-use normalization is semantic. It preserves a tool action across request, stream, tool result, and final answer.

The words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY show the strength of each rule.

## Normative rules

### Compatibility matrix

Phase 1 MUST include these contract rows:

| Harness | Primary protocol | Required fixture |
|---|---|---|
| Claude Code | Anthropic Messages | One streamed tool round trip |
| Codex | OpenAI Responses | One streamed tool round trip |
| OpenCode | OpenAI Chat Completions | One streamed tool round trip |
| Pi | OpenAI Chat Completions | One streamed tool round trip |

The contract manifest MUST contain the exact harness version, protocol version, capture date, and fixture schema version.

The static Phase 1 fixtures are target contracts. Their presence and parse tests do not prove that the current runtime supports them.

Runtime support begins only after the applicable adapter replays the exact fixture. The replay MUST match expected `agent-ir/v1` semantics and native output.

A replay proves compatibility only for its exact harness version and protocol contract. A new harness version MUST pass capture and replay before support expands.

The Phase 1 exit gate contains only four successful streamed tool round trips. It contains one target contract for each harness in the matrix.

Phase 1 does not require failure, parallel-call, cancellation, refusal, or malformed-output fixtures.

Agent IR schema coverage means that a state is representable. It does not prove adapter or runtime support for that state.

### Fixture contents

Each harness fixture MUST contain a sanitized native request and native stream. It MUST also contain the normalized Agent IR representation.

Each fixture MUST show this complete sequence:

- The harness sends instructions, a user message, and tool definitions.
- The provider stream returns one tool call.
- The harness sends the tool result with the same tool-call identifier.
- The provider stream returns the final answer.

The successful fixture MUST preserve event order, finish reason, and usage data when the native protocol supplies them.

Fixture data MUST satisfy ADR 0007. Captures MUST use false credentials, generic paths, and non-personal content.

### Semantic tool-use normalization

A normalized tool call MUST contain `call_id`, `name`, `arguments_raw`, optional `arguments_json`, and sequence data.

A normalized tool result MUST contain the same `call_id`, ordered content, and `is_error`. Empty content and missing content MUST stay different.

ppmlx MUST keep the call identifier across all adapter boundaries. It MUST NOT replace a native identifier after it sends that identifier to a harness.

If a source supplies no identifier, the adapter to Agent IR MUST create one before the first tool-call event.

The identifier MUST stay stable for the full round trip.

An adapter MUST collect decoded argument fragments in source order. It MUST NOT change their character sequence.

`arguments_raw` MUST stay authoritative after stream completion.

The adapter MUST set `arguments_json` only when the complete raw value is valid JSON. It MUST validate that value against the selected tool schema.

The four Phase 1 fixtures prove only one successful tool call and one successful result for each harness target.

Agent IR can represent multiple calls, parallel groups, failed tool results, refusals, cancellations, and provider errors. These states remain unproven.

Text before or after a tool call, a length stop, or another uncovered state also remains unproven until an exact fixture covers it.

ppmlx MUST NOT change a tool name without an explicit alias rule. It MUST NOT guess a tool call from ordinary prose.

The adapter MAY accept a documented provider wrapper, JSON envelope, or model template. Each accepted form MUST have a named, versioned normalization profile.

Malformed or ambiguous tool output MUST produce a typed tool-contract error. ppmlx MUST NOT execute or forward an uncertain tool call.

This malformed-output rule is a target contract. ppmlx does not claim runtime support until the `malformed-tool-output-v1` fixture and replay test pass.

### Future compatibility fixtures

Support for a failure state begins only after its named fixture and exact adapter replay pass.

The future protocol and execution fixture names are:

- `provider-error-v1` for provider and transport failures.
- `tool-result-error-v1` for a failed harness tool result.
- `parallel-tool-calls-v1` for parallel call identity and order.
- `cancellation-v1` for user and harness cancellation.

The future validation and state fixture names are:

- `refusal-v1` for provider refusal semantics.
- `malformed-tool-output-v1` for invalid or ambiguous tool output.
- `stream-state-error-v1` for invalid event order and terminal errors.
- `compatibility-error-v1` for an unsupported required capability.

These fixtures are not part of the Phase 1 exit gate. A future phase can add them without changing the four successful target contracts.

Until a named fixture passes, documentation MUST label the related state as `unproven`. It MUST NOT claim harness, adapter, or runtime support.

### Protocol mappings

Anthropic `tool_use` and `tool_result` blocks MUST map to the same Agent IR semantics as OpenAI tool calls and tool results.

OpenAI Responses function calls and function call outputs MUST keep their call identifiers and item order.

OpenAI Chat Completions tool-call deltas MUST combine by choice index and tool-call index. The adapter MUST keep name and argument fragments in order.

Native protocol fields with no Agent IR meaning MAY pass through in namespaced `extensions`. They MUST NOT change normalized tool semantics.

Hidden reasoning, signatures, and encrypted thinking blocks MUST NOT enter tool arguments, memory, logs, or harness-visible text.

### Stream state

Each adapter MUST use a state machine for streamed output. Each event MUST have a monotonic `sequence` value.

The adapter MUST reject an event that is not valid in the current state.

A tool call MUST start before its argument deltas and end before its tool result. A final response MUST occur after all tool calls reach a terminal state.

An adapter MUST emit a terminal completion or terminal error. A stream error MUST keep the last valid `sequence` value.

The adapter MUST NOT end a stream without a terminal state.

These stream-state rules are target contracts. Phase 1 proves only the successful terminal sequence in each manifest fixture.

### Capability and degradation

Each adapter MUST declare supported protocol features. The router in ADR 0005 MUST use this declaration during candidate selection.

ppmlx MUST NOT silently remove tools, images, structured output, parallel-call meaning, or stream events.

ppmlx permits only these degradation classes:

- `optional_extension_omission` can omit a named optional field from namespaced `extensions`.
- `usage_detail_reduction` can reduce provider usage detail but MUST identify an estimate or aggregate value.
- `stream_chunk_coalescing` can join adjacent text deltas without a character or semantic change.

Each permitted degradation MUST have explicit user policy. The policy MUST name the route, harness protocol, class, and affected field.

The adapter MUST declare a machine-readable method that tells the harness about the degradation.

If the harness cannot receive that notice, ppmlx MUST reject the request before provider use. User policy alone MUST NOT permit silent degradation.

No degradation can remove or change a tool, tool schema, tool-call identifier, image, structured-output rule, stream mode, event order, or terminal state.

`stream_chunk_coalescing` MUST preserve decoded character order, content boundaries, tool-call boundaries, call identifiers, and all terminal events.

An unsupported required feature MUST cause a typed compatibility error before the provider request.

This compatibility-error path remains `unproven` until the `compatibility-error-v1` fixture and replay test pass.

## Security and privacy

Adapters MUST treat tool output and provider stream data as untrusted. They MUST validate structure, size, identifier references, and tool schema before use.

Normalization MUST NOT treat text from a model as a higher-priority instruction. It can classify a tool call only through an accepted profile.

Fixtures and replay output MUST pass credential and private-data scans. A failed scan MUST fail the contract test.

## Consequences

ppmlx can support model-specific formats from Grok, Kimi, DeepSeek, and Qwen without changing harness behavior. Each format needs a tested normalization profile.

Exact fixture versions make support claims narrow and clear. A harness update needs a new capture, replay, and review.

Phase 1 can complete before runtime adapter support exists. Documentation and release notes must keep target contracts separate from verified runtime support.

The target semantic checks reject output that another proxy tries to repair. Runtime support follows the future replay gates in this ADR.

## Rejected alternatives

### Convert all traffic through one OpenAI-shaped JSON object

This conversion loses native stream states and protocol meaning. It can also lose tool-call identifiers and error details.

### Use regular expressions to find tool calls in all model text

Ordinary text can look like a tool call. A generic expression cannot prove tool intent or valid arguments.

### Claim support for all versions of a harness

Harness behavior can change without a protocol version change. Exact version fixtures give a reproducible boundary.

### Fix malformed tool output silently

Silent repair can select the wrong tool or arguments. A named profile can accept known forms, but uncertain output must fail.

### Permit any user-selected degradation

Some losses break tool execution or stream state. Explicit policy cannot make identity, image, tool, or terminal-state loss safe.

## Compatibility effects

Claude Code keeps Anthropic Messages semantics at its ppmlx boundary. Codex keeps OpenAI Responses semantics at its boundary.

OpenCode and Pi keep OpenAI Chat Completions semantics at their boundaries. They share Agent IR meaning, not native wire shape.

Provider adapters can add a model normalization profile without a harness change. A profile change MUST increment its version and replay all affected fixtures.

The existing OpenAI-compatible endpoints remain available. Static Phase 1 fixtures do not change their current runtime support status.

An endpoint can claim harness support only after its adapter replay passes against the exact fixture in the manifest.

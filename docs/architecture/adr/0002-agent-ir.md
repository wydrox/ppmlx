# ADR 0002: Agent IR

- Status: Accepted
- Date: 2026-08-18

## Context

Claude Code, Codex, OpenCode, and Pi do not use one common wire format.
Remote providers also differ in message roles, content blocks, streams, and tool-call formats.

Direct conversion between every harness and provider creates a large conversion matrix.
It also makes identifier loss and stream-order defects difficult to find.

## Decision

ppmlx uses one versioned Agent IR between ingress adapters, the router, and egress adapters.
The Agent IR is lossless for the accepted ppmlx protocol surface.
It keeps source order, stable identifiers, raw tool arguments, and supported vendor data.

The first contract version is `agent-ir/v1`.
The representation contains a conversation envelope, linked requests, ordered blocks, tool definitions, events, usage, and errors.
The normative JSON Schema is [`agent-ir-v1.schema.json`](../schema/agent-ir-v1.schema.json).

## Envelope and request

The top-level envelope contains these fields:

- `ir_version`: The exact Agent IR version.
- `conversation_id`: The required normalized conversation identifier from ADR 0003.
- `source`: The harness name, harness version, protocol, and protocol version.
- `requests`: One initial request and zero or more linked continuation requests.
- `events`: Ordered output, response, and tool lifecycle events.

The envelope can also contain these policy fields:

- `sensitivity`: The optional envelope sensitivity classification.
- `provenance`: The optional envelope provenance record.
- `extensions`: Optional namespaced data that the core does not interpret.

`source` contains `harness`, `harness_version`, `protocol`, and `protocol_version`.
Each request envelope contains `request_id`, `kind`, optional `parent_request_id`, and `request`.
The `kind` value is `initial` or `continuation`.

The first request MUST use `kind=initial` and MUST NOT contain `parent_request_id`.
Each later request MUST use `kind=continuation` and MUST name an earlier request as its parent.
Request identifiers MUST be unique in one conversation.

The `request` object contains these core fields:

- `model`: The requested ppmlx model alias.
- `instructions`: Ordered instruction objects.
- `messages`: Ordered messages with ordered content blocks.

It contains these tool and generation fields:

- `tools`: Ordered tool definitions with `name`, `description`, and `input_schema`.
- `tool_choice`: The normalized tool-selection rule and its source value.
- `generation`: Supported generation controls and explicit source values.
- `stream`: The requested stream mode.

It contains these diagnostic and extension fields:

- `metadata`: Safe caller metadata that affects routing or diagnostics.
- `extensions`: Namespaced source data that the core does not interpret.
- `sensitivity`: The optional request sensitivity classification.
- `provenance`: The optional request provenance record.

## Instruction objects

Each instruction object contains these required fields:

- `source_role`: The native semantic role.
- `source_location`: The JSON Pointer to the native field.
- `order`: The zero-based instruction order.
- `content`: One or more ordered content blocks.

It can also contain `sensitivity`, `provenance`, and `extensions`.
An adapter MUST NOT combine instructions when that action removes role, location, order, or policy data.
If the target needs one instruction string, the egress adapter records the exact composition rule in a versioned profile.

## Continuation requests

A continuation represents a later HTTP request in the same active tool round trip.
Its `parent_request_id` identifies the request that caused the continuation.
The tool result event uses the continuation request identifier.

During an unresolved tool round trip, a continuation MUST repeat these values exactly:

- `model`
- `instructions`
- `tools`
- `tool_choice`
- `generation`
- `stream`

The continuation `messages` contain the full normalized message state for that native request.
They add the related tool results without removal or reorder of prior semantic content.

Existing routing or policy keys in `metadata` MUST keep the same values.
A continuation MAY add request-local diagnostic metadata that does not affect routing, authorization, retention, or provider behavior.

A continuation MAY replace a `native_request` evidence extension with the body for that request.
It MUST keep the same `native_request` evidence keys.
It MAY replace or remove `native_block` evidence when the matching native content block changes.
It MUST keep all semantic and required extensions unchanged.
Any other change causes `continuation_contract_changed` before provider transport starts.

## Content blocks

`agent-ir/v1` defines these common content block kinds:

- `text`
- `image`
- `document`
- `reasoning`

It also defines these action and control block kinds:

- `tool_call`
- `tool_result`
- `refusal`
- `extension`

A `tool_call` contains `call_id`, `name`, `arguments_raw`, and optional `arguments_json`.
A `tool_result` contains `call_id`, ordered result content, and `is_error`.

`arguments_raw` is the exact complete argument text after the ingress adapter decodes the source protocol.
`arguments_json` exists only when the complete text is valid JSON.
The raw value remains authoritative for round-trip conversion.

## Sensitivity and provenance

Content-bearing objects can contain `sensitivity` and `provenance`.
The canonical sensitivity values are `public`, `internal`, `confidential`, and `restricted`.
Their order is `public`, `internal`, `confidential`, and then `restricted` from least to most strict.
If `sensitivity` is absent, ppmlx MUST use `restricted`.

`provenance` contains `origin`, optional `origin_id`, and `trust`.
The canonical origins are `harness`, `provider`, `tool`, `memory`, `ppmlx`, and `unknown`.
The canonical trust values are `trusted`, `untrusted`, and `unknown`.

If source markers are absent, ppmlx MUST use `origin=unknown` and `trust=untrusted`.
The value `trusted` does not grant tool permission and does not increase instruction priority.
Nested content cannot use a less strict sensitivity value than its effective parent value without an explicit policy decision.

## Events

The Agent IR uses ordered events with a unique, monotonic `sequence` value for each `request_id`.
Each event MUST contain the related `request_id`, `choice_index`, and `output_id`.
The envelope supplies `conversation_id`.

Content before, between, or after tool calls uses these lifecycle events:

- `content.started`
- `content.delta`
- `content.completed`

Each content event also contains `content_index`.
The completed event contains the exact normalized content block.

Tool calls and results use these lifecycle events:

- `tool_call.started`
- `tool_call.arguments.delta`
- `tool_call.completed`
- `tool_result`

Tool argument deltas keep their source order.
Each tool event contains `tool_call_index` and the related `call_id`.
Parallel tool events can contain `parallel_group_id`.
The adapter must assemble the deltas without changing their character sequence.
The final tool call uses the same `call_id` as all related events.

`tool_call.started` also contains `name`.
`tool_call.arguments.delta` also contains `delta`.
`tool_call.completed` adds `name`, `arguments_raw`, and optional `arguments_json`.

`tool_result` also contains ordered `content` and `is_error`.
The call ID, choice index, tool index, and parallel group remain stable for one call.
The tool-call lifecycle keeps one output ID.
A tool result can use its own source output ID.

A stream ends with exactly one of these terminal events for each output choice:

- `response.completed`
- `response.refused`
- `response.cancelled`
- `response.failed`

`response.completed` contains `finish_reason` and optional `usage`.
`response.refused` contains a refusal block.
`response.cancelled` contains a reason.
`response.failed` contains an error with `code`, `category`, `message`, and `retryable`.

If a native state has no `agent-ir/v1` representation, the ingress adapter MUST reject it before provider transport starts.

## Native field preservation

Every accepted native request or response field MUST map to a core Agent IR field or a namespaced extension.
An adapter MUST reject an accepted field if it cannot keep the field through both protocol boundaries.

A capture fixture MAY keep a sanitized exact native body in a protocol-namespaced extension.
Examples include `anthropic-messages.native_request` and `openai-responses.native_request`.
This evidence extension MUST NOT contain authorization headers, cookies, tokens, or unsanitized personal data.

A terminal fixture event MAY keep the complete ordered native stream in a `native_stream` extension for its protocol.
Each entry contains the native event name and parsed data value.
This extension must keep terminal sentinels and events with no explicit event name.

The normalized core fields remain authoritative for routing and compatibility checks.
ppmlx MUST NOT use a raw request body as the only semantic representation.

## Normative rules

### Envelope rules

- Every Agent IR object MUST contain `ir_version` or inherit it from its top-level envelope.
- Every object MUST conform to the normative `agent-ir/v1` JSON Schema.
- The first request MUST be `initial`, and every continuation MUST link to an earlier request.
- Every event MUST refer to one request in the same envelope.

### Identity and order

- An adapter MUST preserve message order, block order, event order, and tool order.
- An adapter MUST preserve each source identifier when the source supplies one.
- If the source omits a required identifier, the ingress adapter MUST create one stable identifier.
- A created identifier MUST remain stable for the complete tool round trip.

### Values and extensions

- An adapter MUST keep `arguments_raw` exactly after the source stream completes.
- An adapter MUST NOT repair invalid JSON without an explicit error or policy decision.
- Names in `extensions` MUST use a protocol or provider namespace.
- An egress adapter MUST preserve an extension when the target supports the same meaning.
- The router MUST NOT inspect or change opaque reasoning content.
- An adapter MUST preserve sensitivity and provenance without a silent downgrade.

### Errors and diagnostics

- An adapter MUST reject a required feature that the target cannot represent.
- Unknown required fields MUST cause an error, and unknown optional fields MAY remain in `extensions`.
- Usage values MUST identify their source, and ppmlx MUST NOT present an estimate as a provider value.
- Stream errors MUST end the affected response and keep the last valid sequence value.
- Logs MUST refer to `request_id` and stable call identifiers, not raw private content.
- Schema validation failure MUST stop routing before provider transport starts.

## Security and privacy

The Agent IR can contain source code, file content, images, instructions, and tool results.
It must stay in memory unless a separate retention policy permits storage.

The `extensions` field must not become a path around redaction or route policy.
Adapters must classify sensitive extension data before diagnostics store it.
Opaque reasoning content must not enter logs or memory by default.

Absent classification and provenance data cause the fail-closed defaults.
Adapters must not infer `public` or `trusted` from a message role or a provider name.

The Agent IR must not contain provider credentials.
Authentication profiles use references that the provider transport resolves after routing.

## Consequences

Each harness protocol needs one ingress and one response adapter.
Each provider needs one egress and one response adapter.
The design avoids a converter for each harness-provider pair.

The exact raw and normalized values need more memory during a request.
Phase 1 fixtures prove four successful tool round trips.
Future fixtures must prove parallel calls, mixed content, refusals, cancellations, provider errors, and unsupported-state rejection.

## Compatibility effects

- **Claude Code:** Anthropic `tool_use` and `tool_result` blocks map to stable Agent IR call identifiers.
- **Codex:** Responses API items and function-call argument deltas keep their item order and call identifiers.
- **OpenCode:** Chat Completions messages and `tool_calls` map to ordered Agent IR blocks without identifier loss.
- **Pi:** Chat Completions messages and streamed argument deltas use the same Agent IR rules as OpenCode.

Protocol-specific stop reasons and usage fields stay in namespaced extensions when no common field has the same meaning.
An adapter rejects a request if the target cannot preserve a required harness feature.

## Rejected alternatives

### Pairwise converters

This option adds one converter for every harness-provider pair.
The conversion count grows quickly and produces different behavior for the same source protocol.

### OpenAI Chat Completions as the internal format

This option cannot represent all Responses and Anthropic stream states without private conventions.
It also encourages silent loss of content blocks and identifiers.

### Parsed tool arguments only

This option removes whitespace, duplicate keys, number spelling, and invalid source text.
It prevents a lossless round trip and can change provider behavior.

### Store the complete raw request only

This option preserves data but gives the router no stable semantic contract.
Every provider adapter would still need source-specific logic.

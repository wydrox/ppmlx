# ADR 0009: Bounded Local Tool-Argument Repair

- Status: Proposed
- Date: 2026-08-20
- Supersedes: The blanket rejection of automatic argument repair in ADR 0003
- Clarifies: The Agent IR normalization boundary in ADR 0002

## Context

Local coding models can select the correct tool but emit a small JSON syntax defect in the tool arguments.
A strict parser must reject malformed data.
However, the Phase 4 product contract also permits one deterministic syntax repair.

ADR 0003 rejected automatic argument repair because an unrestricted repair can change user intent, hide provider defects, or break lossless tool transport.
That rejection is correct after ppmlx accepts a tool call into the Agent IR.
It is too broad for a versioned local model-output profile that has not yet produced a valid tool call.

Remote providers can return native structured tool data.
Local models usually return text generated through a tokenizer template or prompt convention.
The two paths need different rules.

## Decision

ppmlx may apply one bounded, deterministic syntax repair to one local model tool-argument payload before it creates an Agent IR tool call.
The repair is part of the versioned local model-output normalization profile.
It is not a general JSON repair service.

The repair does not change tool ownership.
The harness remains the only tool executor.

### Capability levels

Each evaluated model profile declares one tool capability level:

- `native_structured`
- `template_structured`
- `prompt_emulated`
- `none`

A `native_structured` profile must not repair provider tool data.
A `template_structured` or `prompt_emulated` profile can opt in to the repair policy in this ADR.
A `none` profile cannot emit a tool call.

Each evaluated model profile also declares one repair policy:

- `none`
- `single_syntax_v1`

The default policy is `none`.
Existing strict normalization profiles keep their current behavior until an explicit, versioned model profile selects `single_syntax_v1`.

### Normalization boundary

The model-output profile must first recognize an unambiguous tool-call envelope, tool name, and source call identifier when the format supplies one.
It must isolate exactly one argument payload without changing the envelope, tool name, selected tool, or call identifier.

If the profile cannot isolate one argument payload without guessing, normalization fails.
Repair must not make malformed prose look like a tool call.

A successful repair creates the effective argument text for the accepted tool call.
From that point forward, `arguments_raw` is immutable through the Agent IR, protocol adapters, continuation ledger, and harness response path.

### Repair process

The normalizer uses this order:

1. Apply all existing output, call-count, identifier, and argument-size limits.
2. Parse the argument payload without repair.
3. If the payload is one valid JSON object, continue without repair.
4. If strict parsing fails with an eligible syntax condition, select one permitted repair.
5. Apply exactly one transformation.
6. Parse the complete repaired payload with the strict parser.
7. Require one JSON object.
8. Validate the object against the selected tool JSON Schema.
9. Continue only if all checks pass.

The repair budget applies to the complete model output, not to each tool call.
A parallel-call output that needs repairs in two payloads is invalid.
A repair must not invoke the model again.

### Permitted repairs

The `single_syntax_v1` policy permits only these transformations:

#### Unwrap one double-encoded object

The source payload must be one valid JSON string.
The decoded string must contain one complete JSON object and no other non-whitespace data.
The normalizer uses that object text as the repaired payload.

#### Remove one trailing comma

The payload must contain exactly one comma outside a JSON string that is immediately followed by `}` or `]`, with optional whitespace between them.
The normalizer removes that comma only.

#### Append one missing final delimiter

A bounded scanner must prove all these conditions:

- All strings and escape sequences are complete.
- The delimiter stack is valid until the end of the payload.
- Exactly one opening `{` or `[` remains unmatched.
- The required closing delimiter is the final character that is missing.
- No other syntax defect is present.

The normalizer appends that one delimiter only.

If more than one repair rule matches, normalization fails with an ambiguous-repair error.
The implementation must not chain repair rules.

### Prohibited changes

A repair must not:

- add, remove, rename, or select a tool;
- add or change a call identifier;
- invent a key, value, array item, or schema default;
- add quotation marks to a key or string;
- convert single quotes to double quotes;
- change a number, boolean, null value, or Unicode character;
- remove an unknown property to satisfy a schema;
- coerce a scalar to an object or array;
- resolve a duplicate JSON key;
- reorder object properties or array items;
- repair the tool-call envelope;
- use an LLM, network request, provider retry, or probabilistic heuristic.

A missing business value is not a syntax defect.
The normalizer must reject it when the schema requires that value.

### Agent IR representation

For an unrepaired call, Agent IR behavior does not change.

For a repaired call:

- `arguments_raw` contains the complete repaired JSON text.
- `arguments_json` contains the strictly parsed object.
- The completed tool-call event contains a namespaced `ppmlx.tool_argument_repair` extension.

The extension contains only:

- `policy`: `single_syntax_v1`
- `kind`: `double_encoded_object`, `trailing_comma`, or `missing_final_delimiter`
- `profile_id`: the versioned evaluated model-profile identifier
- `source_sha256`: the SHA-256 digest of the malformed argument payload

The extension must not contain the malformed payload, repaired values, prompts, tool results, credentials, or other user content.
The malformed source text remains transient and is discarded after normalization.

### Errors and diagnostics

Repair failures use typed, sanitized errors.
Diagnostics can contain the request identifier, stable call identifier when available, model-profile identifier, repair policy, and error code.
They must not contain raw arguments.

At minimum, the implementation distinguishes these conditions:

- repair not permitted for the selected profile;
- no eligible repair;
- ambiguous repair;
- repair budget exceeded;
- repaired payload is still invalid;
- repaired arguments do not match the tool schema.

Analytics can count repair attempts, successful repairs, repair kinds, and failures.
Analytics must not contain prompts or argument content.

### Evaluation and publication

A repaired call counts as valid only when:

- the strict reparse passes;
- the selected tool exists;
- the JSON Schema validation passes;
- the call identifier remains stable;
- the related tool result links to the correct call.

The capability matrix reports the repair policy and repair rate for each exact evaluated model profile.
It must not present a family-name match as an evaluated model result.

The stable, preview, experimental, and disabled thresholds remain those in the Phase 4 work plan.
Repair can improve the valid-call rate, but the published result must show how often repair was required.

### Security and resource limits

Repair runs after the existing input-size limits and before tool execution.
The implementation must also apply output and JSON limits to the repaired payload.

The repair scanner must have bounded memory use and linear processing time in the payload size.
It must not use recursive parsing before the configured depth checks.
It must not weaken duplicate-key rejection, schema limits, regular-expression time limits, unknown-tool rejection, or continuation identity checks.

Remote native structured tool data remains fail-closed.
ppmlx must not repair it into a different provider response.

## Normative rules

- ppmlx MUST parse without repair first.
- ppmlx MUST apply no more than one deterministic transformation to one payload in one model output.
- ppmlx MUST enable repair only through an explicit versioned evaluated model profile.
- ppmlx MUST NOT repair `native_structured` provider data.
- ppmlx MUST NOT repair a tool-call envelope, tool name, call identifier, or business value.
- ppmlx MUST rerun strict parsing and tool-schema validation after repair.
- ppmlx MUST preserve the repaired `arguments_raw` without later changes.
- ppmlx MUST record sanitized repair provenance for a successful repair.
- ppmlx MUST discard the malformed payload after normalization unless a separate explicit retention policy permits it.
- ppmlx MUST keep tool execution and approval in the harness.

## Consequences

A local model can recover from one narrow syntax defect without giving ppmlx authority to infer user intent.
The Agent IR stays lossless from the accepted normalization boundary forward.
The capability matrix can distinguish reliable native output from output that depends on repair.

The normalizer needs a bounded lexical scanner, versioned repair policy, additional typed errors, and evaluation counters.
Some malformed outputs that appear easy to fix will remain rejected because the repair is not provably unambiguous.

## Rejected alternatives

### Repair every tool call independently

This option gives a parallel output multiple repair attempts.
It can hide broad model instability and makes the one-repair rule ineffective.

### General-purpose tolerant JSON parser

This option can accept many non-standard forms and make implementation behavior library-dependent.
It also makes it difficult to explain what changed.

### Schema-driven value completion

This option invents business values or defaults.
It changes model intent and can cause an unsafe tool action.

### Ask the model to repair its output

This option is probabilistic, adds another model turn, and can change the selected tool or arguments.
It is not deterministic syntax repair.

### Repair remote native structured calls

This option changes provider-owned structured data and hides provider contract defects.
Remote providers must return valid structured calls or a typed error.

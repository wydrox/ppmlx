# ADR 0009: Bounded Local Tool-Argument Repair

- Status: Accepted
- Date: 2026-08-20
- Amends: ADR 0002 and ADR 0003

## Context

Local coding models can emit the correct tool name and values with a small JSON syntax defect.
Examples include one trailing comma, one missing final delimiter, or one JSON object encoded as a JSON string.

Phase 4 requires one deterministic syntax repair before ppmlx rejects the tool call.
ADR 0002 already permits repair after an explicit policy decision.
ADR 0003 rejects automatic argument repair.
An implicit or broad repair can change provider output and hide defects.

ppmlx needs a narrow exception that preserves tool identity and keeps the harness in control.
The exception must fail closed when the intended JSON is not unambiguous.

## Decision

ppmlx uses strict parsing first.
A versioned model tool profile can then permit one bounded repair policy named `bounded-v1`.
The repair is part of provider-output normalization.
It occurs before ppmlx creates a completed Agent IR tool-call event.

The policy applies only when all these conditions are true:

- The selected model tool capability level is `template_structured` or `prompt_emulated`.
- The evaluated model profile explicitly enables `bounded-v1`.
- The complete model output is available.
- The strict parser rejected the argument text.
- The repair result is one valid JSON object.

A `native_structured` provider response is never repaired.
A profile with capability level `none` cannot receive a tool request.

## Capability levels

Each model tool profile uses one capability level:

- `native_structured`: The provider returns protocol-native structured tool data.
  ppmlx does not parse or repair plain text.
- `template_structured`: A verified tokenizer template emits one documented structured text format.
- `prompt_emulated`: A versioned prompt and parser contract emulates structured tool output.
- `none`: The route does not support tool use.

Only an evaluated `template_structured` or `prompt_emulated` profile can enable `bounded-v1`.
The profile must name its parser and repair policy explicitly.
A model-family substring or an inferred template cannot enable repair.

## Repair budget

One complete model output has one shared repair budget.
The budget is not one repair per tool call.

ppmlx can apply at most one transformation to the complete output.
If two parallel calls need repair, ppmlx rejects the output.
If one transformation would need a second transformation, ppmlx rejects the output.
If more than one permitted transformation could produce a result, ppmlx rejects the output as ambiguous.

A successful repair does not permit a retry with a different parser or model profile.

## Permitted transformations

`bounded-v1` permits only these transformations:

### One double-encoded object

The original value must be one valid JSON string.
Decoding that string one time must produce one valid JSON object with no remaining text.

### One trailing comma

The source can contain exactly one comma outside a JSON string immediately before `}` or `]`.
Removing that comma must produce one valid JSON object with no other syntax change.

### One missing final delimiter

All JSON strings must be closed.
All existing delimiters must be correctly nested.
Exactly one final `}` or `]` can be missing.
Appending that one required delimiter must produce one valid JSON object.

## Prohibited changes

The repair engine must not:

- Add or remove a tool call.
- Change the selected tool or tool name.
- Change a source call identifier.
- Add a missing identifier.
- Add, remove, rename, or reorder an argument property.
- Add a schema default or another business value.
- Quote an unquoted property name.
- Convert single quotes to double quotes.
- Change a string, number, Boolean, or null value.
- Accept duplicate JSON keys.
- Repair a truncated string.
- Repair mixed prose and tool output.
- Apply a universal tolerant parser.

## Agent IR boundary

For an unrepaired call, `arguments_raw` remains the exact accepted argument text from the versioned profile.

For a repaired call, `arguments_raw` is the exact repaired JSON text that passed strict parsing and schema validation.
`arguments_json` is parsed from that repaired text.
All emitted argument deltas must assemble to the repaired `arguments_raw` value.
The malformed source text is outside the accepted Agent IR boundary and must not replace `arguments_raw`.

The completed tool-call event records this namespaced extension:

```json
{
  "ppmlx.tool_argument_repair": {
    "policy": "bounded-v1",
    "kind": "trailing_comma",
    "profile": "qwen-json-v1"
  }
}
```

The `kind` value is one of:

- `double_encoded_object`
- `trailing_comma`
- `missing_final_delimiter`

The extension must not contain the malformed arguments, repaired arguments, prompt, tool result, or a credential.
Logs and diagnostics use the request ID, call ID, profile, repair kind, and typed error code only.

This decision keeps Agent IR lossless from the accepted, versioned provider-normalization boundary forward.
It does not claim byte-for-byte preservation of rejected malformed model text.

## Validation order

ppmlx uses this order:

1. Select one explicit model tool profile.
2. Parse the complete output with the strict profile parser.
3. If strict parsing fails, check whether `bounded-v1` is enabled and unused.
4. Apply at most one permitted transformation.
5. Parse the result again with the same strict parser.
6. Reject an unknown tool name or invalid call identifier.
7. Validate the argument object against the selected tool JSON Schema.
8. Create Agent IR events and preserve all accepted identifiers.
9. Return the tool call to the harness.

A repaired value that fails JSON Schema validation is still invalid.
ppmlx must not use the schema to invent or coerce a value.

## Tool ownership and identity

The harness remains the only tool executor.
Repair does not grant ppmlx permission to run a tool.

Repair must preserve the selected tool, tool order, source identifier mapping, and generated `call_id`.
It must also preserve `tool_call_index`, `choice_index`, `output_id`, and optional `parallel_group_id`.
The harness returns the result through the normal continuation contract.

## Errors

A failed repair must return a typed, sanitized error.
The implementation uses these error codes:

- `malformed_arguments`: No permitted repair matches.
- `tool_argument_repair_budget_exceeded`: The output needs more than one transformation.
- `tool_argument_repair_ambiguous`: More than one permitted repair result is possible.
- `arguments_not_object`: The repaired value is not one JSON object.
- `tool_arguments_schema_mismatch`: The repaired object fails the selected tool schema.

Errors must not include model output or argument values.

## Capability and evaluation effects

Each evaluated model profile declares:

- The capability level.
- The normalization profile.
- Whether `bounded-v1` is enabled.
- The model and tokenizer revisions.
- The repair count and repair rate.
- The valid-call rate across three fixed evaluation runs.

A capability score includes repaired calls only when the repair followed this ADR.
The published capability matrix must show repair use separately from strict-valid calls.
A family-name match alone does not establish a stable capability claim.

## Security and privacy

The repair engine operates on the same bounded argument text that the strict parser receives.
Existing byte, depth, node, string, and call-count limits apply before and after repair.

The engine must not persist malformed source text.
Analytics must not receive prompts, arguments, repair payloads, or raw model reasoning.
The retention and redaction rules in ADR 0007 still apply.

## Consequences

A small, documented class of local model syntax defects can complete a tool round trip.
The behavior is deterministic, profile-specific, testable, and visible in capability results.

Some outputs that a permissive parser could guess remain rejected.
This is intentional.
The implementation needs dedicated repair tests and evaluation counters before Phase 4 can close.

## Rejected alternatives

### No repair

This option keeps the old strict behavior but does not satisfy the Phase 4 product contract.

### Universal tolerant JSON parser

This option accepts an open-ended set of malformed outputs.
It makes behavior difficult to test and can change business values.

### Repair native structured provider data

This option hides a remote provider protocol defect.
Native structured provider output must fail through the provider adapter instead.

### Schema-driven completion

This option uses defaults or inferred values to satisfy a tool schema.
It invents business values and can cause an unintended tool action.

### Multiple repair attempts

This option chains transformations until parsing succeeds.
It makes the accepted result depend on repair order and creates ambiguity.

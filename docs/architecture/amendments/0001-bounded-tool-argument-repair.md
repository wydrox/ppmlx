# Architecture Amendment 0001: Bounded Tool-Argument Repair

- Status: Accepted
- Date: 2026-08-20
- Amends: ADR 0003
- Clarifies: ADR 0002
- Implementation status: Contract only. ppmlx 0.9.1 remains strict.

## Context

Some local models emit a documented tool-call envelope with malformed JSON arguments.
A harness cannot run that call safely until the arguments are valid JSON and match the selected tool schema.

ADR 0002 requires an explicit error or policy decision before ppmlx changes invalid JSON.
ADR 0003 rejected automatic argument repair because hidden or open-ended repair can change intent and break lossless transport.

Phase 4 requires one deterministic syntax repair.
The repair must not become a general parser, a business-value generator, or a way to hide model defects.

## Decision

ppmlx can apply one bounded JSON syntax repair at the versioned local model-profile boundary.
The repair is optional for each profile and disabled unless the profile enables it explicitly.

The repair boundary is before Agent IR accepts the tool call.
After this boundary, `arguments_raw` is the exact argument text that ppmlx validates, emits, and sends to the harness.
ppmlx does not change that accepted text later in the request.

The malformed model output is provider evidence, not accepted Agent IR content.
ppmlx records sanitized repair metadata that contains no argument-derived content or digest.
It does not store the malformed argument text for this purpose.

## Capability levels

Each model profile declares one tool capability level:

- `native_structured`: The provider returns a structured call through a native API.
- `template_structured`: A verified tokenizer template emits a documented call envelope.
- `prompt_emulated`: A documented prompt profile asks a model to emit a call envelope.
- `none`: The profile does not support tool calls.

Only `template_structured` and `prompt_emulated` profiles can enable bounded repair.
A `native_structured` profile must return a typed error for malformed argument JSON.
A profile with level `none` must not receive a tool-use route.

A capability level does not prove quality.
The published capability matrix must also identify the exact model, tokenizer revision, evaluation result, and support status.

## Repair budget

One complete model output has one repair budget.
The budget permits at most one deterministic transformation.

ppmlx must first parse the output without repair.
It can attempt repair only after strict parsing returns a repair-eligible syntax error.
It must then run the normal strict parser again.

A repaired output must still pass all normal checks:

- output and argument size limits;
- duplicate-key rejection;
- tool-name selection;
- tool-call identifier rules;
- parallel-call rules;
- JSON Schema validation;
- continuation and result-link validation.

If the second strict parse or any later check fails, ppmlx must reject the call.
It must not attempt another repair.

For parallel calls, the single budget applies to the complete output.
If more than one call needs repair, ppmlx must reject the output.

## Permitted transformations

The first repair policy is `bounded-json-v1`.
It permits only these transformations:

1. **One double-encoded object unwrap.**
   The original argument value must be one valid JSON string.
   Decoding that string must produce one complete JSON object.

2. **One trailing-comma removal.**
   Exactly one comma outside a JSON string can be removed when it occurs immediately before `}` or `]`.

3. **One final-delimiter append.**
   The scanner can append one missing final `}` or `]` only when all strings are closed, all earlier delimiters are valid, and exactly one container remains open.

Each transformation must have one canonical implementation and one stable repair code.
The same input, profile, limits, and policy must produce the same result.

## Prohibited changes

A repair must not:

- change the selected tool;
- create, remove, or change a call identifier;
- add a missing business value;
- add a schema default;
- quote an unquoted key;
- convert single quotes to double quotes;
- change a string, number, Boolean, null, property name, or array order;
- remove or choose between duplicate keys;
- infer a tool call from ordinary assistant prose;
- combine two transformations;
- repair a tool result;
- execute a tool.

If a permitted transformation has more than one possible edit location, ppmlx must reject the output as ambiguous.

## Agent IR representation

A repaired tool call uses the normal Agent IR tool-call lifecycle.
The accepted, repaired text becomes `arguments_raw`.
The parsed object becomes `arguments_json`.

The related completed tool-call event includes this namespaced extension:

```json
{
  "ppmlx.tool_argument_repair": {
    "policy": "bounded-json-v1",
    "kind": "trailing_comma",
    "profile": "qwen-json-v1"
  }
}
```

The extension must not include raw model output, raw arguments, argument-derived hashes, secrets, prompts, or tool results.
A harness must never need the extension to link a result to the call.

Adapters must preserve this extension when the target protocol supports equivalent metadata.
Otherwise, the extension remains internal diagnostic data.

## Errors and diagnostics

The normalizer must return typed, sanitized errors.
The error must identify whether repair was unavailable, ineligible, ambiguous, exhausted, or unsuccessful.

Logs and traces can contain:

- request, output, and call identifiers;
- model profile and capability level;
- repair policy and repair kind;
- sanitized outcome or error code.

They must not contain the malformed argument text, a repaired value that can include secrets, or an argument-derived hash.

Profile evaluation must report repair counts and repair rates.
A high repair rate can prevent a profile from receiving stable status even when its final valid-call rate passes the numeric threshold.

## Security and privacy

Model output is untrusted.
Repair does not increase its trust or sensitivity classification.

The repair implementation must use bounded scanning and parsing.
It must obey the same byte, depth, node, string, call-count, and time limits as strict normalization.
It must not resolve external schema references or fetch remote data.
It must not create a durable content fingerprint from malformed or repaired arguments.
ADR 0007 prohibits an unsalted hash of a credential because it can still identify a low-entropy secret.

The harness remains the only tool executor.
Repair does not grant tool permission and does not bypass harness approval.

## Compatibility effects

- Existing valid local tool calls remain unchanged.
- Existing local profiles keep strict behavior until they enable `bounded-json-v1` explicitly.
- Native structured remote profiles remain strict.
- Existing call identifiers and continuation rules remain unchanged.
- Existing local commands and non-tool requests remain unchanged.

## Consequences

A narrow class of local model syntax defects can become valid, auditable tool calls.
The model profile, not a universal parser, owns the decision to permit repair.

The implementation needs repair-specific fixtures, security tests, evaluation metrics, and capability-matrix fields.
A repaired call is not evidence that the original model output was valid.

## Rejected alternatives

### Hidden best-effort repair

This option changes arguments without an explicit profile or audit record.
It hides provider defects and makes behavior difficult to reproduce.

### Repeated repair attempts

This option chains transformations until parsing succeeds.
It expands the interpretation surface and can invent intent.

### Schema-driven completion

This option fills missing values from defaults, descriptions, or examples.
It invents business values that the model did not supply.

### Universal plain-text parser

This option searches ordinary prose for tool-like JSON.
It can turn descriptive text into an unintended tool call.

### Repair inside protocol adapters

This option gives each harness protocol different repair behavior.
It recreates pairwise conversion rules and weakens the common Agent IR boundary.

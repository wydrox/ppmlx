# ADR 0009: Bounded Local Tool-Argument Repair

- Status: Accepted
- Date: 2026-08-20
- Amends: ADR 0002
- Partially supersedes: ADR 0003, `Automatic argument repair`

## Context

Local models can emit a documented tool-call format with a small JSON syntax defect.
The existing local runtime rejects malformed arguments.
ADR 0002 permits repair only after an explicit policy decision.
ADR 0003 rejects automatic repair because an unrestricted repair can change user intent and hide provider defects.

The Phase 4 product contract requires one deterministic syntax repair.
It also requires PPMLX to preserve tool names, call identifiers, result links, and harness tool ownership.
The repair policy must therefore be narrow, visible, testable, and local to a versioned model profile.

Remote providers can return native structured tool calls.
PPMLX must use that structure directly.
It must not parse or repair plain text when native structured data exists.

## Decision

PPMLX may apply one bounded syntax repair to local model tool arguments.
The repair occurs inside a versioned local model-output profile.
It occurs after PPMLX has received the complete tool-argument payload and before PPMLX emits the related Agent IR tool-call events.

PPMLX must try the strict parser first.
It may start repair only when the strict parser rejects the argument payload or when the decoded payload is one JSON string instead of one JSON object.

One complete model output has one shared repair budget.
A successful transformation consumes that budget.
PPMLX must not chain two transformations.
For parallel calls, PPMLX may repair at most one argument payload in the complete output.
If two calls need repair, PPMLX must reject the output.

The same input, model profile, limits, and repair policy must always produce the same accepted output or the same error.
Repair must not depend on a model call, network access, time, locale, random value, or external state.

## Capability levels

Each model tool profile must declare one capability level:

- `native_structured`: The provider supplies structured tool data. Text repair is not permitted.
- `template_structured`: A verified tokenizer template supplies a documented text grammar. A named repair policy may be permitted.
- `prompt_emulated`: A versioned prompt supplies a documented text grammar. A named repair policy may be permitted.
- `none`: The route does not support tool calls.

A profile must declare its normalization grammar and repair policy separately.
A family-name match does not prove support for a model checkpoint.
Published support applies to an evaluated model repository, model revision, tokenizer revision, and quantization.

## Permitted transformations

A repair policy may enable any subset of the following transformations.
The profile must name each enabled transformation.

### Unwrap one encoded object

PPMLX may unwrap one JSON string when all these conditions are true:

1. The original argument payload is valid JSON.
2. The decoded value is a string.
3. The string contains exactly one valid JSON object.
4. No data exists before or after that object.

PPMLX must not apply another repair after the unwrap.

### Remove one trailing comma

PPMLX may remove one comma when all these conditions are true:

1. The comma is inside the identified argument payload.
2. The comma is outside a JSON string.
3. Only whitespace occurs between the comma and the next `}` or `]`.
4. The complete argument payload contains no second repairable trailing comma.
5. Removing the comma makes the payload valid JSON.

### Append one final closing delimiter

PPMLX may append one `}` or `]` when all these conditions are true:

1. The argument payload has valid string and escape boundaries.
2. The payload has exactly one unmatched opening `{` or `[`.
3. The unmatched opening delimiter is the last structural defect.
4. Only whitespace occurs after the final complete token.
5. Appending the matching delimiter makes the payload valid JSON.

## Prohibited changes

PPMLX must not use repair to:

- add a missing business value;
- add a schema default;
- quote an unquoted property name;
- convert single quotes to double quotes;
- change a string, number, Boolean, or null value;
- rename, add, remove, or reorder a property;
- change the selected tool;
- change or create a source call identifier;
- accept a duplicate JSON key;
- guess an argument boundary;
- repair the model-profile envelope outside the identified argument payload;
- repair a partial stream chunk;
- repair remote `native_structured` output;
- apply a second transformation.

If the versioned profile cannot identify the exact argument payload without guessing, PPMLX must reject the output.

## Validation order

PPMLX must use this order:

1. Select one explicit model tool profile.
2. Identify the complete argument payload through that profile grammar.
3. Apply the strict JSON parser.
4. When permitted, apply one deterministic repair.
5. Apply the strict JSON parser again.
6. Require a JSON object.
7. Reject duplicate keys and non-finite values.
8. Validate resource limits.
9. Validate the tool name and call identity.
10. Validate the arguments against the selected JSON Schema.
11. Emit Agent IR events.
12. Return the tool call to the harness.

A repaired payload that fails the tool schema is invalid.
PPMLX must not use schema information to invent a value or perform a second repair.

The harness remains the tool owner.
This ADR does not give PPMLX permission to execute a tool.

## Agent IR representation

For a call that does not need repair, `arguments_raw` keeps the exact accepted source argument text.

For a repaired call, `arguments_raw` contains the complete repaired argument text that passed strict parsing and schema validation.
This value becomes the effective argument payload for the tool round trip.
`arguments_json` contains its parsed JSON value.

The related Agent IR object must include this namespaced extension:

```json
{
  "ppmlx.tool_argument_repair": {
    "kind": "remove_single_trailing_comma",
    "profile": "qwen-json-v1"
  }
}
```

The `kind` value must name the one applied transformation.
The `profile` value must name the exact versioned normalization profile.

PPMLX must not place the malformed source payload in Agent IR, logs, traces, analytics, snapshots, or durable memory.
It may keep the payload in process memory only for the active normalization operation.
It must discard that payload after it emits the accepted result or error.

Agent IR remains lossless from the accepted normalization boundary forward.
This ADR does not claim byte preservation for malformed provider text that the policy rejects and replaces before Agent IR event creation.

## Identifiers and streaming

Repair applies only to argument syntax.
PPMLX must preserve the tool name, source call identifier, call order, tool-call index, output identifier, parallel group, and result link.

PPMLX must not emit `tool_call.started` for a repaired call until it knows the tool name and the stable PPMLX call identifier.
It must not emit any argument delta from malformed source text.
For a repaired call, the emitted argument delta sequence must reconstruct the effective `arguments_raw` value exactly.

A model-output profile may buffer one complete local tool turn before it emits external stream events.
It must document that behavior and must not claim live token streaming for the buffered turn.

## Errors and diagnostics

A failed repair must return a typed normalization error.
The error may include the profile name, repair kind, request identifier, and call identifier.
It must not include raw model output or raw tool arguments.

Diagnostics may count:

- strict parse success;
- repair attempts;
- repair success by kind;
- repair rejection by reason;
- schema rejection after repair.

Diagnostics must not contain prompts, raw arguments, tool results, provider credentials, or model reasoning.

## Evaluation and publication

Deterministic fixtures for every enabled transformation must pass at 100%.
Tool-call and result correlation must pass at 100%.

Each evaluated model profile must report:

- model repository and revision;
- tokenizer revision;
- quantization;
- normalization profile;
- capability level;
- repair policy;
- case-set version;
- three fixed evaluation runs;
- strict valid-call rate;
- accepted valid-call rate after repair;
- repair count and repair rate;
- call and result correlation rate;
- support classification.

Support classifications use these thresholds:

- `stable`: At least 98% accepted valid calls across three fixed runs.
- `preview`: 95% to 97.9% accepted valid calls.
- `experimental` or `disabled`: Below 95%.

The capability matrix must show strict and repaired results separately.
A high repair rate must remain visible.
PPMLX must not use repair to present an unstable profile as natively reliable.

## Security and privacy

Tool arguments are untrusted model output.
The strict parser, size limits, depth limits, node limits, duplicate-key checks, and bounded schema validator still apply after repair.

A repair implementation must be pure and resource-bounded.
It must scan strings and escape sequences explicitly.
It must not use regular expressions that can cause unbounded backtracking on model output.

The repair extension is diagnostic metadata.
It does not grant trust, tool permission, or a lower sensitivity classification.

## Consequences

Local profiles can recover from a small and verified syntax defect.
They cannot use a general JSON fixer.
Remote structured provider data stays strict.
The capability matrix can distinguish strict reliability from repaired reliability.

The Agent IR schema does not need a new version.
The existing namespaced `extensions` field can record the repair decision.

Implementations need additional fixtures, profile metadata, evaluation reports, and security tests.
The current Phase 4 runtime remains unchanged until those implementation gates pass.

## Rejected alternatives

### No repair

This option keeps the old strict behavior.
It does not satisfy the Phase 4 product contract for one deterministic repair.

### Universal JSON repair

This option applies a general fixer to all models and formats.
It can change meaning, hide profile defects, and make evaluation results difficult to reproduce.

### Model-based repair

This option sends malformed arguments to another model.
It is nondeterministic and can invent values.

### Schema-driven completion

This option fills missing values from defaults, descriptions, or inferred user intent.
It changes business data and exceeds syntax repair.

### Repair native structured provider data

This option ignores the provider's structured contract.
A native provider defect must remain an explicit provider error.

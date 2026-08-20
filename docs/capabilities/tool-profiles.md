# Local Tool Profile Capability Matrix

This document separates parser support from measured model capability.

A normalization profile tells ppmlx how to read one documented tool-call format.
It does not prove that every model with a similar family name produces reliable calls.
PPMLX publishes a model capability only for an exact model revision, tokenizer revision, quantization, and evaluated environment.

The machine-readable matrix is [`tool-profiles.json`](tool-profiles.json).

## Current published status

| Normalization profile | Capability level | Repair policy | Published support | Evaluated exact models |
|---|---|---|---|---:|
| `grok-openai-chat-v1` | `template_structured` | none | disabled | 0 |
| `kimi-k2-v1` | `template_structured` | none | disabled | 0 |
| `deepseek-v3-v1` | `template_structured` | none | disabled | 0 |
| `qwen-json-v1` | `template_structured` | none | disabled | 0 |

The runtime has strict parsers for these formats.
It also contains a bounded repair engine behind profile metadata.
No shipped profile enables repair, and no exact model profile has completed the publication gate.

## Publication gate

A published profile must use the fixed case set in [`cases-v1.json`](../../tests/fixtures/tool_profile_eval/cases-v1.json).
It must complete three runs on macOS on Apple Silicon.

The gate is:

- Deterministic parser and repair fixtures: 100%.
- Tool-call and result correlation: 100%.
- Stable: at least 98% effective valid calls in each run.
- Preview: 95% to 97.9% effective valid calls in each run.
- Below 95%: experimental.
- Any fixture or correlation failure: disabled.

The matrix shows strict validity and repair use separately.
A repaired call counts as effective only after strict reparse, tool-name checks, JSON Schema validation, and call correlation pass.

## Required evidence

Each report records:

- PPMLX version and commit.
- Model repository and immutable model revision.
- Immutable tokenizer revision.
- Quantization.
- Normalization profile and capability level.
- Repair policy.
- Apple chip, memory, and macOS version.
- Generation settings and fixed seed schedule.
- Case-set version.
- Each run result and aggregate result.
- Strict-valid, repaired-valid, effective-valid, and correlation rates.

Reports do not contain prompts, generated model text, tool arguments, tool results, secrets, or raw errors.

## Run an evaluation

Run this command from a clean ppmlx checkout on an Apple Silicon Mac:

```bash
uv run python scripts/evaluate_tool_profiles.py \
  --model mlx-community/EXACT-MODEL \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --tokenizer-revision 89abcdef0123456789abcdef0123456789abcdef \
  --quantization 4bit \
  --profile qwen-json-v1 \
  --capability-level template_structured \
  --repair-policy none \
  --output docs/capabilities/evaluations/example.json
```

The runner refuses non-Apple-Silicon systems, dirty tracked files, mutable revision labels, nonzero temperature, and non-JSON output paths.
It always runs the fixed case set three times.

## Publication rules

1. Commit the generated report under `docs/capabilities/evaluations/`.
2. Review the exact model, tokenizer, environment, and generation metadata.
3. Verify all deterministic fixtures and normal CI gates.
4. Add only that exact model profile to the JSON matrix.
5. Enable a runtime repair policy only when the exact evaluated profile passes its gate.
6. Do not copy one checkpoint score to another checkpoint, tokenizer, or quantization.
7. Do not publish a family-level score.

A later evaluation can lower or disable support.
The matrix must show the latest accepted evidence for each exact model profile.

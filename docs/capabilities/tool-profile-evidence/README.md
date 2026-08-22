# Local tool-profile evidence

This directory contains reviewed, machine-readable reports for exact model profiles.

A report is publishable only when it:

- uses `tool-profile-report/v1`;
- identifies immutable PPMLX, model, and tokenizer commits;
- records the exact quantization and normalization profile;
- records one macOS Apple Silicon environment;
- uses the fixed case set and three unique seeds;
- carries a recorded deterministic fixture artifact for the exact PPMLX commit;
- contains no prompts, generated model text, tool arguments, tool results, secrets, or raw reasoning;
- passes the evidence validator and the full repository quality suite.

Publication fails closed without that fixture artifact. The
`deterministic_fixtures_passed` field alone is not evidence.

Generate a report from a clean checkout on an Apple Silicon Mac:

```bash
uv run python scripts/run_deterministic_fixtures.py \
  --output docs/capabilities/tool-profile-evidence/fixtures.json

uv run python scripts/evaluate_local_tool_profile.py \
  --model-path /absolute/path/to/exact-model-checkout \
  --model-repository mlx-community/EXACT-MODEL \
  --model-revision 0123456789abcdef0123456789abcdef01234567 \
  --tokenizer-revision 89abcdef0123456789abcdef0123456789abcdef \
  --quantization 4bit \
  --normalization-profile qwen-json-v1 \
  --capability-level template_structured \
  --repair-policy none \
  --fixtures-evidence docs/capabilities/tool-profile-evidence/fixtures.json \
  --output docs/capabilities/tool-profile-evidence/example.json
```

Then regenerate the public matrix:

```bash
uv run python scripts/render_tool_capability_matrix.py \
  --fixtures-evidence docs/capabilities/tool-profile-evidence/fixtures.json
uv run python scripts/render_tool_capability_matrix.py \
  --fixtures-evidence docs/capabilities/tool-profile-evidence/fixtures.json --check
```

The repository currently ships no evaluated exact model profile and enables no repair policy in production. Do not copy a score between checkpoints, tokenizer revisions, or quantizations. Do not publish a model-family score.

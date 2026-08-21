# Local tool capability matrix

This matrix contains exact reviewed model profiles. A parser profile without reviewed evidence is not a model capability claim.

| Model | Revision | Format | Level | Repair | Strict | Effective | Correlation | Status |
|---|---|---|---|---|---:|---:|---:|---|
| — | — | `grok-openai-chat-v1` | `template_structured` | `none` | — | — | — | **not_evaluated** |
| — | — | `kimi-k2-v1` | `template_structured` | `none` | — | — | — | **not_evaluated** |
| — | — | `deepseek-v3-v1` | `template_structured` | `none` | — | — | — | **not_evaluated** |
| — | — | `qwen-json-v1` | `template_structured` | `none` | — | — | — | **not_evaluated** |

## Publication gates

- Deterministic parser and correlation fixtures must pass at 100%.
- Publication fails closed without a recorded fixture artifact for the exact ppmlx commit; the flag alone is not evidence.
- Each exact model profile must complete three fixed runs.
- Stable requires at least 98% effective valid calls in every run and no more than 2% repaired valid calls in any run.
- Preview requires at least 95% effective valid calls in every run.
- Lower results are experimental. Any fixture or correlation failure disables the profile.
- Family-name matching does not create a capability claim. A model identifier selects a profile only when it names one exact reviewed repository.

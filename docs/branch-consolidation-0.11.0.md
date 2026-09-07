# Branch consolidation for 0.11.0

Date: 2026-09-07

This change combines the local and fetched remote branch histories in `main`. The repository has one registered worktree.

## Source decisions

- The experimental branch supplies the local agent, voice support, prompt cache, draft decoding, RAG, document processing, templates, gateway, batch queue, and API playground.
- The policy router remains in `ppmlx/router.py`. Optional `model:auto` selection uses `ppmlx/auto_router.py`.
- Current strict runtime checks, credential storage, memory read guards, download progress, and the dynamic model registry remain in place.
- Commit `774abb9` supplies the worker ownership fix. Worker-scoped failure also requires a claimed job. Regression tests cover stale ownership and completed jobs.
- Commit `b8025df` supplies the commit-bound fixture evidence tests.
- The Gemma streaming and safe error changes from `dev` are already present in the current code and tests.
- The current canonical ADR 0009 supersedes the older repair drafts. Its tests define the current repair rules, including a single missing envelope brace.
- Current provider, profile evaluation, security, and Homebrew files supersede their older branch copies. Old release version edits do not replace version 0.11.0.
- The README now documents both the current proxy and the optional local agent. The proxy does not execute harness tools. The explicit `ppmlx agent` command can execute tools.

The experimental source was merged with conflict resolution. After the missing fixes and tests were ported, a history merge retained the reviewed current tree for the remaining branch tips. This avoids restoring superseded implementations or old version values.

## Remaining branch tips included by the history merge

| Branch | Tip |
| --- | --- |
| `agent/update-readme-router` | `50fe93c` |
| `proxy/phase-1-contracts` | `4521bc0` |
| `proxy/release-0.6.0` | `426d263` |
| `origin/dev` | `f56a090` |
| `origin/fix/extraction-claim-race` | `774abb9` |
| `origin/fix/normalization-gates` | `b8025df` |
| `origin/proxy/homebrew-release-recovery` | `51035b9` |
| `origin/proxy/phase-4-bounded-json-core` | `fc771b1` |
| `origin/proxy/phase-4-bounded-json-integration` | `d277848` |
| `origin/proxy/phase-4-normalization-contract` | `caeee93` |
| `origin/proxy/phase-4-normalization-contract-reviewed` | `52e1f0a` |
| `origin/proxy/phase-4-tool-profile-evaluation` | `2107066` |
| `origin/proxy/phase-4-tool-profile-evaluation-reviewed` | `0a787ad` |
| `origin/proxy/phase-5-provider-interface` | `a71a26d` |
| `origin/proxy/release-0.9.1` | `77dd006` |
| `origin/proxy/security-privacy-docs` | `35b9ecc` |

## Validation

- Python test suite: 1,582 passed.
- Ruff: passed for package, tests, and scripts.
- Mypy: passed for all 95 package source files.
- Wheel, source distribution, metadata, and package content checks: passed.

These checks use the test suite's MLX fixtures. They do not establish model output quality or live GPU performance. This source version does not publish a package to PyPI.

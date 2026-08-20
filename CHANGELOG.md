# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.9.0] - 2026-08-20

### Added
- An opt-in, buffered Agent IR runtime for streamed tool turns on Chat Completions, Responses HTTP, and Anthropic Messages.
- Versioned local output profiles for Grok, Kimi K2, DeepSeek V3, and Qwen tool-call formats.
- A scoped continuation ledger that keeps call identity, route pins, retries, parallel groups, expiry, and tombstones without tool arguments or results.

### Changed
- The MLX engine can now reject a tokenizer that drops tool definitions instead of silently retrying without tools.
- Request body limits now count received HTTP chunks when `Content-Length` is absent.
- The strict runtime now applies parallel-tool, usage, storage, and Claude Code thinking options before local generation.

### Security
- Validate local tool arguments against a bounded JSON Schema subset and apply time limits to string patterns.
- Reject strict-runtime tool traffic that could fall back to a legacy path, including Responses WebSocket tool requests.
- Restrict the opt-in strict runtime to loopback clients and isolate continuation state by principal, project, harness, conversation, and call.
- Bound local generation, output normalization, active conversations, retained Agent IR bytes, and replay responses.
- Apply hard continuation-ledger backpressure, keep credential values out of scope identifiers, and join exact concurrent retries without a second generation.

## [0.8.0] - 2026-08-18

### Added
- Pure protocol facades for Claude Code 2.1.231, Codex 0.147.0, OpenCode 1.18.18, and Pi 0.84.2.
- Exact request, tool-result, and streamed-response replay through the `agent-ir/v1` contract fixtures.

### Security
- Reject credentials, duplicate JSON keys, invalid tool links, unsafe stream frames, and inputs that exceed adapter limits.
- Keep native protocol evidence off by default and remove private validation data from public adapter errors.

## [0.7.0] - 2026-08-18

### Added
- Strict typed models and lossless JSON helpers for the `agent-ir/v1` request, content, tool, and event contract.
- Agent IR validation for continuation links, stable tool calls, event order, terminal states, sensitivity, and protocol extensions.

### Changed
- Require Pydantic 2.5 or later for strict Agent IR JSON value types.

## [0.6.0] - 2026-08-18

### Added
- Accepted architecture decisions for the local model router, lossless tool-use normalization, provider authentication, routing, memory, retention, and harness compatibility.
- Sanitized streamed tool-use contract fixtures for Claude Code, Codex, OpenCode, and Pi.

## [0.5.9] - 2026-08-18

### Added
- Python 3.11 and 3.12 quality gates with Ruff, strict Mypy, and clean package-install tests.
- Exact TestPyPI and PyPI artifact verification, an Apple Silicon generation test, and verified GitHub Release creation.

### Changed
- Restrict the memory MCP dependency to the compatible MCP 1.x API.
- Make memory writes, retrieval, conflict migration, and atom compaction use exact project, app, and session namespaces.

### Fixed
- Correct vision routing and tokenizer access across Chat Completions, Responses, and WebSocket APIs.
- Preserve Responses API image input and fix Anthropic stream request logging.
- Keep additive preferences, constraints, and todos active unless a correction names the prior value.

### Security
- Redact common provider credentials before events, jobs, candidates, graph data, atoms, and FTS data reach SQLite.
- Prevent reused or incomplete session identifiers from retrieving memory from another project or app.

## [0.5.8] - 2026-07-29

### Added
- Hybrid memory search ranking with lexical, semantic, dense/hash, type, recency, and namespace signals plus explicit `score` output.
- Temporal single-value supersede policy for mutable state slots (`current_task`, latest commits/status, preferences/decisions).
- Memory maintenance APIs/CLI: `set_fact`, `fact_history`, `temporal-conflicts`, `migrate-temporal-conflicts`, `doctor`, `expire`, `embed-cache`, `compact-atoms`.
- Deterministic offline hash embedding cache for candidate re-ranking without downloading an embedding model.
- Durable `memory_atoms` compaction from high-confidence active candidates, with handoff/context preference for atoms.

### Changed
- Workflow-state write path is no longer fully additive; only explicit history predicates remain append-only.
- Noisy namespace detection now also covers smoke/e2e test namespaces.
- Memory search supports `--project/--session/--app`, excludes noisy namespaces by default, and surfaces rank components in deep mode.

### Fixed
- CLI `memory temporal-conflicts` / `migrate-temporal-conflicts` no longer crash due to missing store methods.

## [0.5.7] - 2026-07-09

### Changed
- Move repository metadata and install links from `the-focus-company/ppmlx` to `wydrox/ppmlx` after the GitHub transfer.

## [0.5.6] - 2026-05-18

### Added
- Configurable HuggingFace registry auto-refresh for the pull/model picker list via `[registry].refresh` and `PPMLX_REGISTRY_REFRESH`.
- Registry refresh setting in the interactive config TUI.
- Manual registry refresh from model pickers with `r`, plus `ppmlx pull --refresh` for a forced refresh before selection.

### Fixed
- Added coverage for registry cache refresh behavior and registry refresh config parsing.

## [0.5.4] - 2026-05-07

### Fixed
- Strip Gemma 4 channel-style thought markers (`<|channel>thought ... <channel|>`) from visible model output while preserving the final answer.
- Treat Gemma channel thought markers as reasoning markers in streaming Anthropic and Responses API output.

## [0.5.3] - 2026-05-07

### Added
- Model-aware process titles for `ppmlx run <model>` and `ppmlx serve <model>`.
- Safe CLI/API error tracking for analytics.

### Changed
- Updated analytics configuration to the current PostHog project.
- Require `mlx-lm>=0.31.3` for Gemma 4 model support.

### Fixed
- Gemma 4 loading via the dynamic registry no longer fails with unsupported `gemma4` model type.
- Anthropic `/v1/messages` streaming no longer consumes MLX generation from a background thread, avoiding MLX thread-local stream crashes.
- Anthropic tool/agent requests disable thinking so responses surface as visible text/tool output instead of hidden reasoning only.
- Plain model output is no longer incorrectly treated as hidden thinking when the model does not start inside a `<think>` block.

## [0.4.2] - 2026-04-01

### Added
- TurboQuant section in README linking to ppmlx.dev/turboquant
- Website moved to dedicated repo (the-focus-company/ppmlx.dev)

## [0.4.1] - 2026-03-31

### Changed
- Deduplicate `_resolve_model_path` across engine modules into `models.py`
- Extract shared think-tag stream processor, eliminating ~100 lines of duplication
- Remove `setproctitle` dependency

### Fixed
- Incorrect `reasoning_text` assignment in streaming responses
- `_flush_port` now verifies PID belongs to ppmlx before killing (H3)
- Vision engine rejects `file://` URLs and bare paths from API requests (C3)

### Security
- CORS defaults to localhost-only; configurable via `cors_origins` in config.toml (C2)
- Request body size limit middleware (default 10 MB, configurable) (H1)
- Server-side `max_tokens` cap (default 32768, configurable) (H2)
- Embedding input limited to 256 texts per request (H4)
- WebSocket message size limit (10 MB) (H5)
- Removed debug JSONL logging to `/tmp/` (C1)

### Added
- `SECURITY_AUDIT.md` documenting all findings and fixes
- Homebrew formula with `arch: :arm64` constraint and auto-update workflow

## [0.4.0] - 2026-03-30

### Added
- Thinking/reasoning model support: `think` and `reasoning_budget` API parameters
- `reasoning_effort` mapping (low/medium/high) to reasoning budget tokens
- Thinking metrics tracking in SQLite DB with migration
- Streaming thinking/reasoning delta support in chat completions
- Empty-answer retry logic for thinking models in engine
- `ppmlx logs` and `ppmlx stats` CLI commands for log analysis
- `ppmlx config --thinking`, `--reasoning-budget`, `--effort-base`, `--max-tools-tokens` flags
- `[thinking]` section in config (`enabled`, `default_reasoning_budget`, `effort_base`)
- Thinking configuration panel in TUI

## [0.3.0] - 2026-03-28

### Added
- First-run analytics opt-in prompt (analytics disabled by default)
- Configurable CORS origins via `PPMLX_CORS_ORIGINS` env var
- Pydantic validation on all API request bodies (bounds checking, batch limits)
- Interactive Swagger docs at `/docs` and ReDoc at `/redoc`
- Network binding warning when server exposed on non-localhost
- Version sync test (pyproject.toml vs __init__.py)
- ruff linter and mypy type checker in CI pipeline
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md
- GitHub issue templates (bug report, feature request) and PR template
- "Requirements" and "ppmlx vs Ollama" sections in README

### Changed
- Analytics default changed from opt-out to opt-in
- API error responses now return generic messages (no internal details leaked)
- Removed `allow_credentials=True` from CORS middleware
- `uv.lock` now tracked in git (removed from .gitignore)

### Fixed
- Unused variables and imports flagged by ruff

## [0.2.0] - 2026-03-27

### Added
- Analytics module with privacy-first design (opt-in, data sanitization, DNT support)
- First-run prompt asking users to opt in to anonymous analytics
- Curses-based TUI model picker with search/filter
- Open WebUI launcher support
- Responses API endpoint (`/v1/responses`) for Codex compatibility
- Anthropic Messages API endpoint (`/v1/messages`)
- Vision model support via mlx-vlm
- Model quantization command (`ppmlx quantize`)
- SQLite request logging and metrics (`/metrics` endpoint)
- Tool calling support with awareness injection
- Configurable tool awareness prompts
- Interactive model selection for serve/run/rm commands
- CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md

### Changed
- Expanded core tool list with case-insensitive matching
- Improved streaming with thinking model support (`<think>` blocks)
- Generic error messages in API responses (no internal details leaked)
- Pydantic validation on all API request bodies

### Removed
- Debug request logging to `/tmp`

## [0.1.0] - 2026-03-20

### Added
- Initial release
- CLI with serve, pull, run, list, ps, rm, config commands
- OpenAI-compatible API server (chat completions, completions, embeddings)
- Model registry with 168+ pre-configured models
- Homebrew formula
- Astro marketing website

# ppmlx

**Run LLMs on your Mac.** OpenAI-compatible API powered by Apple Silicon.

[![CI](https://github.com/wydrox/ppmlx/actions/workflows/tests.yml/badge.svg)](https://github.com/wydrox/ppmlx/actions)
[![PyPI](https://img.shields.io/pypi/v/ppmlx)](https://pypi.org/project/ppmlx/)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

## Install

```bash
uv tool install ppmlx
```

> Requires macOS on Apple Silicon (M1+) and Python 3.11+
>
> Privacy note: `ppmlx` never sends prompts, responses, file contents, paths, or tokens anywhere. Optional anonymous usage analytics can be disabled with `ppmlx config --no-analytics`.

## Get Started

```bash
ppmlx pull qwen3.5:9b      # download a model
ppmlx run qwen3.5:9b       # chat in the terminal
ppmlx serve                 # start API server on :6767
```

### curl | sh (one-liner)

```bash
curl -fsSL https://raw.githubusercontent.com/wydrox/ppmlx/main/scripts/install.sh | sh
```

### From source

```bash
git clone https://github.com/wydrox/ppmlx
cd ppmlx
uv tool install .
```

### Homebrew

Homebrew tap coming soon. For now, use `uv tool install ppmlx`.

---

## Quick Start

```bash
# 1. Download a model
ppmlx pull llama3

# 2. Interactive chat REPL
ppmlx run llama3

# 3. Start OpenAI-compatible API server on :6767
ppmlx serve
```

---

## Benchmarks

Measured on a MacBook Pro M4 Pro (48 GB unified memory, macOS 15.x). Each scenario was run 3 times with `temperature=0` and `max_tokens=8192`; values below are averages.

### GLM-4.7-Flash (4-bit, ~5 GB)

| Scenario | Metric | ppmlx | Ollama | Delta |
|---|---|---|---|---|
| **Simple** (short prompt, short answer) | tok/s | 63.1 | 40.5 | **+56%** |
| | TTFT | 374 ms | 832 ms | **-55%** |
| **Complex** (short prompt, long answer) | tok/s | 55.6 | 38.8 | **+43%** |
| | TTFT | 496 ms | 412 ms | +20% |
| **Long context** (~4 K token prompt) | tok/s | 42.1 | 27.5 | **+53%** |
| | TTFT | 6,792 ms | 8,401 ms | **-19%** |

### Qwen 3.5 9B (4-bit, ~6 GB)

| Scenario | Metric | ppmlx | Ollama | Delta |
|---|---|---|---|---|
| **Simple** | tok/s | 48.2 | 22.7 | **+112%** |
| | TTFT | 537 ms | 324 ms | +66% |
| **Complex** | tok/s | 47.2 | 23.0 | **+106%** |
| | TTFT | 567 ms | 455 ms | +25% |
| **Long context** | tok/s | 43.2 | 23.7 | **+82%** |
| | TTFT | 9,212 ms | 11,461 ms | **-20%** |

> **tok/s** = tokens per second (higher is better). **TTFT** = time to first token (lower is better). Delta is relative to Ollama.

**Methodology.** Streaming chat completions over the OpenAI-compatible API; TTFT measured from request start to first SSE content chunk. See [`scripts/bench_common.sh`](scripts/bench_common.sh) and the per-model scripts in `scripts/` for the full, reproducible setup.

That's it. Any OpenAI-compatible tool works out of the box:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:6767/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3.5:9b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Commands

| Command | Description | Key Options |
|---|---|---|
| `ppmlx launch` | Interactive launcher (pick action + model) | `-m model`, `--host`, `--port`, `--flush` |
| `ppmlx serve` | Start API server on :6767 | `-m model`, `--embed-model`, `-i`, `--no-cors` |
| `ppmlx run <model>` | Interactive chat REPL | `-s system`, `-t temp`, `--max-tokens` |
| `ppmlx pull [model]` | Download model (multiselect if no arg) | `--token` |
| `ppmlx list` | Show downloaded models | `-a` all (incl. registry), `--path` |
| `ppmlx rm <model>` | Remove a model | `-f` skip confirmation |
| `ppmlx ps` | Show loaded models & memory | |
| `ppmlx quantize <model>` | Convert & quantize HF model to MLX | `-b bits`, `--group-size`, `-o output` |
| `ppmlx graph` | Print a temporal memory graph snapshot as JSON | `--project`, `--session`, `--query`, `--status` |
| `ppmlx memory status/search/list/handoff/compact-stats` | Inspect the experimental local temporal memory graph | `--json`, `--status`, `--scope`, `--session` |
| `ppmlx memory jobs/worker/rebuild/prune` | Manage async extraction jobs and graph maintenance | `--status`, `--once`, `--max-jobs`, `--dry-run` |
| `ppmlx memory-eval` | Run the anti-garbage memory eval suite | `--json`, `--dataset`, `--predictions` |
| `ppmlx compact-eval` | Run long-session rolling-context compaction evals | `--json`, `--output` |
| `ppmlx answer-quality-eval` | Score compact-answer quality across recall, wrong facts, actionability, grounding, and A/B equivalence | `--json`, `--dataset`, `--template` |
| `ppmlx answer-quality-replay` | Run real Pi/Claude session quality eval through a live local ppmlx server | `--model`, `--source`, `--base-url` |
| `ppmlx quality-bench` | Split a real long session into 80% prefix / 20% holdout probes and compare local answers to recorded answers | `--split`, `--max-probes`, `--model` |
| `ppmlx trace export` / `ppmlx compact-replay` | Export and replay local traces through compact mode | `--project`, `--session`, `--expect` |
| `ppmlx config` | View/set configuration | `--hf-token` |

## Connect Your Tools

Point any OpenAI-compatible client at `http://localhost:6767/v1` with any API key:

- **Cursor** — Settings > AI > OpenAI-compatible
- **Continue** — config.json: provider `openai`, apiBase above
- **LangChain / LlamaIndex** — set `base_url` and `api_key="local"`

## Config

Optional. `~/.ppmlx/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 6767

[defaults]
temperature = 0.7
max_tokens = 2048

[analytics]
enabled = true
provider = "posthog"
respect_do_not_track = true
```

### Experimental local memory

Shadow-mode memory capture stores request/response events and high-precision memory candidates locally in `~/.ppmlx/memory.db`. It does **not** inject memory into prompts yet.

```toml
[memory]
enabled = true
mode = "shadow"   # off | shadow | compact | inject
# compact mode keeps a rolling prompt tail and renders scoped graph context
rolling_tokens = 10000
hot_tail_tokens = 6500
session_context_tokens = 2000
compact_threshold_tokens = 12000
max_context_items = 40

# graph-memory extraction
# default rule_based extraction runs synchronously; set extractor="model_memory_json" to enqueue async jobs
extractor = "rule_based"      # rule_based | model_memory_json (llm_json/gemma_json are legacy aliases)
extraction_model = "gemma-4-e2b"
extraction_workers = 1
extraction_max_tokens = 1200       # output tokens per extraction call
extraction_input_tokens = 6000     # approximate input budget per chunk
extraction_overlap_tokens = 600    # overlap between chunks for cross-boundary facts
extraction_max_chunks_per_event = 32
extraction_timeout_seconds = 45
```

Modes:
- `shadow`: store events/candidates only; prompts are unchanged.
- `compact`: before inference, replace long histories with system context from the graph + a hot tail.
- `inject`: reserved for compact + broader memory retrieval.

Graph-engine maintenance is local and explicit: `model_memory_json` extraction is asynchronous via durable jobs processed by `ppmlx memory worker`; the default `rule_based` extractor remains synchronous. `llm_json` and `gemma_json` are still accepted as legacy aliases. Long events are split into token-budgeted extraction chunks with overlap before model extraction.

Compact observability is recorded locally in `memory.db` and, if analytics are enabled, sent as privacy-safe aggregate metrics to PostHog. It never sends prompts, responses, tool output, model repo IDs, project IDs, or session IDs.

Tool/MCP outputs are distilled through a plugin-style distiller interface. The built-in generic JSON distiller extracts small evidence-backed atoms such as candidates, prices, availability, specs, source URLs, and rejected items, while raw JSON stays local in the event log.

CLI:

```bash
ppmlx memory config --enabled --extractor model_memory_json --model gemma-4-e2b
ppmlx memory config --input-limit 6000 --overlap 600
ppmlx memory status
ppmlx memory search "concise answers"
ppmlx memory list --status active
ppmlx memory handoff --project tv-shopping --session tv-session-001
ppmlx memory compact-stats --since 24
ppmlx memory jobs --status pending
ppmlx memory worker --once
ppmlx memory rebuild --dry-run
ppmlx memory prune --dry-run
ppmlx graph --project tv-shopping --session tv-session-001 > graph.json
ppmlx trace export --project tv-shopping --session tv-session-001 --output trace.json
ppmlx compact-replay trace.json --expect "budget = 5000 PLN"
ppmlx memory-eval
ppmlx compact-eval
ppmlx answer-quality-eval
ppmlx answer-quality-replay ~/.pi/agent/sessions/.../session.jsonl \
  --model mlx-community/Qwopus3.5-4B-v3-4bit \
  --base-url http://127.0.0.1:6767/v1
ppmlx quality-bench ~/.pi/agent/sessions/.../session.jsonl \
  --split 0.8 --max-probes 5 \
  --model mlx-community/Qwopus3.5-4B-v3-4bit
```

`ppmlx graph` prints a local graph snapshot as JSON. The browser-based graph viewer has been removed; memory data remains local in `memory.db`.

`answer-quality-replay` requires a running local ppmlx server. It generates a compact answer and a local reference answer, selects question-relevant required facts, filters embedded examples/fixtures, and reports recall, wrong facts, actionability, grounding, and A/B equivalence.

`quality-bench` is the stronger quality benchmark: it splits a real transcript by episodes into prefix and held-out suffix, feeds only the compacted prefix plus held-out user turn to the local model, and scores the response against the recorded next assistant answer.

`trace export` is local-only and may include prompts, responses, and tool outputs. Keep exported traces private unless you intentionally want to share them.

## Anonymous Usage Analytics

`ppmlx` supports privacy-preserving anonymous product analytics, disabled by default. On first interactive run, the beta onboarding asks whether you want to help by enabling it.

What is sent:
- command and API event names such as `serve_started`, `model_pulled`, `api_chat_completions`
- app version, Python minor version, OS family, CPU architecture
- a random anonymous install id, used only to count returning beta installs
- coarse booleans/counters such as `stream=true`, `tools=true`, `batch_size=4`

What is never sent:
- prompts, responses, tool arguments, file contents, file paths
- HuggingFace tokens, API keys, repo IDs, model prompts, request bodies

When events are sent:
- when a CLI command starts
- when OpenAI-compatible API endpoints are hit

Why:
- understand which workflows matter most during beta
- prioritize compatibility work across commands and API surfaces
- measure adoption without collecting user content

Opt out:

```bash
ppmlx config --no-analytics
```

or:

```toml
[analytics]
enabled = false
```

By default, opted-in beta analytics are sent to the maintainer-operated PostHog project. To use your own PostHog sink instead, configure:

```bash
export PPMLX_ANALYTICS_HOST="https://analytics.example.com"
export PPMLX_ANALYTICS_PROJECT_API_KEY="your-posthog-project-api-key"
```

If you prefer, you can also set the same values in `~/.ppmlx/config.toml`.

## API Documentation

When the server is running, interactive API docs are available at:

- **Swagger UI**: [http://localhost:6767/docs](http://localhost:6767/docs)
- **ReDoc**: [http://localhost:6767/redoc](http://localhost:6767/redoc)

## Requirements

- macOS on Apple Silicon (M1 or later)
- Python 3.11+
- At least 8 GB unified memory (16 GB+ recommended for larger models)

## ppmlx vs Ollama

| | ppmlx | Ollama |
|---|---|---|
| Runtime | MLX (Apple-native) | llama.cpp (cross-platform) |
| Platform | macOS Apple Silicon only | macOS, Linux, Windows |
| GPU backend | Metal (unified memory) | Metal / CUDA / ROCm |
| API | OpenAI-compatible | Ollama + OpenAI-compatible |
| Language | Python | Go + C++ |
| Quantization | MLX format | GGUF format |

Choose **ppmlx** if you want maximum Apple Silicon performance with a pure-Python, MLX-native stack. Choose **Ollama** if you need cross-platform support or GGUF models.

## License

MIT

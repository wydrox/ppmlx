"""Built-in benchmarking for ppmlx models.

Runs standardized scenarios against the local ppmlx server and collects
TTFT, tokens/sec, total latency, prompt tokens, and completion tokens.
"""
from __future__ import annotations

import json
import math
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table


# ── Scenario prompts (mirrors bench_common.sh S1-S3) ─────────────────

SIMPLE_PROMPT = (
    "Explain the difference between TCP and UDP in exactly 3 bullet points. "
    "Be precise and technical."
)

COMPLEX_PROMPT = (
    "Design a rate limiter for an API gateway. Describe:\n"
    "1. The algorithm you'd choose and why "
    "(token bucket vs sliding window vs fixed window)\n"
    "2. Data structures needed\n"
    "3. How to handle distributed deployment (multiple gateway instances)\n"
    "4. Edge cases (clock skew, burst traffic, graceful degradation)\n"
    "Provide pseudocode for the core logic."
)

# ~4K tokens of filler for the long-context scenario.  We build it
# programmatically so there are no huge string literals to maintain.
_LONG_CONTEXT_FILLER = "\n".join(
    f"def function_{i}(x_{i}: int) -> int:\n"
    f'    """Compute step {i} of the pipeline."""\n'
    f"    return x_{i} * {i + 1} + {i * 7}\n"
    for i in range(200)
)

LONG_CONTEXT_PROMPT = (
    "Analyze the following Python module. Provide:\n"
    "1. A summary of its architecture and main classes\n"
    "2. Thread-safety analysis — are there race conditions?\n"
    "3. Performance bottlenecks and optimization suggestions\n"
    "4. Any bugs or edge cases that could cause failures\n\n"
    f"```python\n{_LONG_CONTEXT_FILLER}\n```"
)

SCENARIOS: dict[str, dict[str, Any]] = {
    "simple": {
        "label": "Simple",
        "description": "Short prompt -> short answer (baseline throughput + TTFT)",
        "prompt": SIMPLE_PROMPT,
        "max_tokens": 256,
    },
    "complex": {
        "label": "Complex",
        "description": "Short prompt -> long answer (~2K tokens, sustained generation)",
        "prompt": COMPLEX_PROMPT,
        "max_tokens": 2048,
    },
    "long_context": {
        "label": "LongContext",
        "description": "~4K token prompt -> medium answer (prefill speed)",
        "prompt": LONG_CONTEXT_PROMPT,
        "max_tokens": 512,
    },
}


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class IterationResult:
    """Metrics from a single benchmark iteration."""
    ttft_ms: float = 0.0
    tokens_per_sec: float = 0.0
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


@dataclass
class ScenarioStats:
    """Aggregated statistics for a scenario across N iterations."""
    scenario: str
    label: str
    iterations: list[IterationResult] = field(default_factory=list)

    @property
    def successful(self) -> list[IterationResult]:
        return [r for r in self.iterations if r.error is None]

    def _values(self, attr: str) -> list[float]:
        return [getattr(r, attr) for r in self.successful]

    @staticmethod
    def _stat(values: list[float]) -> dict[str, float]:
        if not values:
            return {"avg": 0.0, "stddev": 0.0, "min": 0.0, "max": 0.0}
        n = len(values)
        avg = sum(values) / n
        # Bessel's correction (n-1) for sample stddev; fall back to n=1
        variance = sum((v - avg) ** 2 for v in values) / max(1, n - 1)
        return {
            "avg": round(avg, 2),
            "stddev": round(math.sqrt(variance), 2),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
        }

    def stats(self) -> dict[str, dict[str, float]]:
        return {
            "ttft_ms": self._stat(self._values("ttft_ms")),
            "tokens_per_sec": self._stat(self._values("tokens_per_sec")),
            "total_latency_ms": self._stat(self._values("total_latency_ms")),
            "prompt_tokens": self._stat(self._values("prompt_tokens")),
            "completion_tokens": self._stat(self._values("completion_tokens")),
        }


@dataclass
class BenchmarkResult:
    """Complete benchmark result for one model."""
    model: str
    timestamp: str
    runs: int
    system_info: dict[str, Any]
    scenarios: dict[str, ScenarioStats] = field(default_factory=dict)
    draft_model: str | None = None  # set when speculative decoding was used

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "model": self.model,
            "timestamp": self.timestamp,
            "runs": self.runs,
            "system_info": self.system_info,
            "results": {},
        }
        if self.draft_model:
            out["draft_model"] = self.draft_model
        for name, ss in self.scenarios.items():
            out["results"][name] = {
                "label": ss.label,
                "stats": ss.stats(),
                "iterations": [
                    {
                        "ttft_ms": r.ttft_ms,
                        "tokens_per_sec": r.tokens_per_sec,
                        "total_latency_ms": r.total_latency_ms,
                        "prompt_tokens": r.prompt_tokens,
                        "completion_tokens": r.completion_tokens,
                        "error": r.error,
                    }
                    for r in ss.iterations
                ],
            }
        return out


# ── BenchmarkRunner ──────────────────────────────────────────────────

class BenchmarkRunner:
    """Runs standardized benchmark scenarios against a ppmlx server."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:6767",
        runs: int = 3,
        scenarios: list[str] | None = None,
        timeout: float = 120.0,
        draft_model: str | None = None,
        speculative_tokens: int | None = None,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.runs = runs
        self.scenario_names = scenarios or list(SCENARIOS.keys())
        self.timeout = timeout
        self.draft_model = draft_model
        self.speculative_tokens = speculative_tokens
        self._console = Console()

        # Validate scenario names
        invalid = set(self.scenario_names) - set(SCENARIOS.keys())
        if invalid:
            raise ValueError(
                f"Unknown scenarios: {', '.join(sorted(invalid))}. "
                f"Available: {', '.join(SCENARIOS.keys())}"
            )

    def _run_single(self, scenario_name: str, client: httpx.Client) -> IterationResult:
        """Execute one iteration of a scenario via streaming API call."""
        scenario = SCENARIOS[scenario_name]
        url = f"{self.base_url}/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": scenario["prompt"]}],
            "max_tokens": scenario["max_tokens"],
            "temperature": 0.0,
            "stream": True,
        }
        if self.draft_model:
            payload["draft_model"] = self.draft_model
        if self.speculative_tokens is not None:
            payload["speculative_tokens"] = self.speculative_tokens

        start = time.monotonic()
        ttft: float | None = None
        completion_tokens = 0
        prompt_tokens = 0

        try:
            with client.stream(
                "POST", url, json=payload,
                headers={"Authorization": "Bearer local"},
            ) as resp:
                resp.raise_for_status()
                buf = ""
                for raw_chunk in resp.iter_text():
                    buf += raw_chunk
                    while "\n\n" in buf:
                        line, buf = buf.split("\n\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(data_str)
                            delta = obj.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if ttft is None:
                                    ttft = (time.monotonic() - start) * 1000
                                completion_tokens += 1
                            usage = obj.get("usage")
                            if usage:
                                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        except (json.JSONDecodeError, IndexError, KeyError):
                            pass
        except httpx.HTTPStatusError as exc:
            return IterationResult(error=f"HTTP {exc.response.status_code}: {exc.response.text[:200]}")
        except httpx.ConnectError:
            return IterationResult(error="Connection refused — is ppmlx server running?")
        except Exception as exc:
            return IterationResult(error=str(exc)[:200])

        elapsed_ms = (time.monotonic() - start) * 1000
        tps = (completion_tokens / (elapsed_ms / 1000)) if elapsed_ms > 0 and completion_tokens > 0 else 0.0

        return IterationResult(
            ttft_ms=round(ttft or 0.0, 2),
            tokens_per_sec=round(tps, 2),
            total_latency_ms=round(elapsed_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _get_system_info(self) -> dict[str, Any]:
        """Collect basic system info for the results file."""
        info: dict[str, Any] = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }
        try:
            from ppmlx.memory import get_system_ram_gb
            info["ram_gb"] = round(get_system_ram_gb(), 1)
        except Exception:
            pass
        return info

    def run(self) -> BenchmarkResult:
        """Execute all scenarios for the configured number of runs."""
        result = BenchmarkResult(
            model=self.model,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            runs=self.runs,
            system_info=self._get_system_info(),
            draft_model=self.draft_model,
        )

        for name in self.scenario_names:
            result.scenarios[name] = ScenarioStats(
                scenario=name,
                label=SCENARIOS[name]["label"],
            )

        with httpx.Client(timeout=self.timeout) as client, Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=self._console,
        ) as progress:
            total = self.runs * len(self.scenario_names)
            task = progress.add_task("Benchmarking...", total=total)

            for run_idx in range(self.runs):
                for name in self.scenario_names:
                    label = SCENARIOS[name]["label"]
                    progress.update(
                        task,
                        description=f"Run {run_idx + 1}/{self.runs} - {label}",
                    )
                    iteration = self._run_single(name, client)
                    result.scenarios[name].iterations.append(iteration)
                    progress.advance(task)

        return result


# ── Display helpers ──────────────────────────────────────────────────

def print_results(result: BenchmarkResult, console: Console | None = None) -> None:
    """Print benchmark results as a Rich table."""
    console = console or Console()

    title = f"Benchmark Results: {result.model}"
    if result.draft_model:
        title += f" (draft: {result.draft_model})"
    table = Table(title=title)
    table.add_column("Scenario", style="cyan", no_wrap=True)
    table.add_column("TTFT (ms)", justify="right")
    table.add_column("Tok/s", justify="right", style="green")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Prompt Tok", justify="right", style="dim")
    table.add_column("Compl Tok", justify="right", style="dim")

    for name, ss in result.scenarios.items():
        stats = ss.stats()
        errors = len(ss.iterations) - len(ss.successful)
        error_note = f" ({errors} err)" if errors else ""

        ttft = stats["ttft_ms"]
        tps = stats["tokens_per_sec"]
        lat = stats["total_latency_ms"]
        pt = stats["prompt_tokens"]
        ct = stats["completion_tokens"]

        table.add_row(
            f"{ss.label}{error_note}",
            f"{ttft['avg']:.0f} +/-{ttft['stddev']:.0f}",
            f"{tps['avg']:.1f} +/-{tps['stddev']:.1f}",
            f"{lat['avg']:.0f} +/-{lat['stddev']:.0f}",
            f"{pt['avg']:.0f}",
            f"{ct['avg']:.0f}",
        )

    console.print(table)
    info_parts = [
        f"Runs: {result.runs}",
        f"System: {result.system_info.get('platform', 'unknown')}",
        f"RAM: {result.system_info.get('ram_gb', '?')} GB",
    ]
    if result.draft_model:
        info_parts.append(f"Speculative: {result.draft_model}")
    console.print(f"\n[dim]{' | '.join(info_parts)}[/dim]")


_COMPARISON_METRICS = [
    ("ttft_ms", "ms", False),
    ("tokens_per_sec", "tok/s", True),
    ("total_latency_ms", "ms", False),
]


def _print_comparison_table(
    a: BenchmarkResult,
    b: BenchmarkResult,
    *,
    title: str,
    col_a: str,
    col_b: str,
    col_delta: str,
    show_multiplier: bool = False,
    console: Console | None = None,
) -> None:
    """Shared helper for side-by-side benchmark comparison tables."""
    console = console or Console()

    table = Table(title=title)
    table.add_column("Scenario", style="cyan", no_wrap=True)
    table.add_column("Metric", style="dim")
    table.add_column(col_a, justify="right")
    table.add_column(col_b, justify="right")
    table.add_column(col_delta, justify="right")

    for name in a.scenarios:
        if name not in b.scenarios:
            continue
        a_stats = a.scenarios[name].stats()
        b_stats = b.scenarios[name].stats()
        label = a.scenarios[name].label

        for metric, unit, higher_is_better in _COMPARISON_METRICS:
            a_val = a_stats[metric]["avg"]
            b_val = b_stats[metric]["avg"]

            pct = ((b_val - a_val) / a_val * 100) if a_val > 0 else 0.0
            improved = (pct > 0 and higher_is_better) or (pct < 0 and not higher_is_better)
            color = "green" if improved else "red" if pct != 0 else "dim"
            sign = "+" if pct > 0 else ""

            delta_text = f"[{color}]{sign}{pct:.1f}%[/{color}]"
            if show_multiplier and metric == "tokens_per_sec" and a_val > 0:
                delta_text += f" [{color}]({b_val / a_val:.2f}x)[/{color}]"

            table.add_row(label, f"{metric} ({unit})", f"{a_val:.1f}", f"{b_val:.1f}", delta_text)
            label = ""

    console.print(table)


def print_speculative_comparison(
    baseline: BenchmarkResult,
    speculative: BenchmarkResult,
    console: Console | None = None,
) -> None:
    """Print side-by-side comparison: normal vs speculative decoding."""
    draft_label = speculative.draft_model or "speculative"
    _print_comparison_table(
        baseline, speculative,
        title=f"Speculative Decoding: {baseline.model}\nnormal vs draft={draft_label}",
        col_a="Normal", col_b="Speculative", col_delta="Speedup",
        show_multiplier=True, console=console,
    )


def print_comparison(
    current: BenchmarkResult,
    baseline: BenchmarkResult,
    console: Console | None = None,
) -> None:
    """Print side-by-side comparison between current and baseline results."""
    _print_comparison_table(
        baseline, current,
        title=f"Comparison: {current.model} vs baseline",
        col_a="Baseline", col_b="Current", col_delta="Delta",
        console=console,
    )


def save_results(result: BenchmarkResult, path: Path) -> Path:
    """Write benchmark results to a JSON file. Returns the path written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    return path


def load_results(path: Path) -> BenchmarkResult:
    """Load benchmark results from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    result = BenchmarkResult(
        model=data["model"],
        timestamp=data["timestamp"],
        runs=data["runs"],
        system_info=data.get("system_info", {}),
        draft_model=data.get("draft_model"),
    )

    for name, scenario_data in data.get("results", {}).items():
        ss = ScenarioStats(
            scenario=name,
            label=scenario_data.get("label", name),
        )
        for it_data in scenario_data.get("iterations", []):
            ss.iterations.append(IterationResult(
                ttft_ms=it_data.get("ttft_ms", 0.0),
                tokens_per_sec=it_data.get("tokens_per_sec", 0.0),
                total_latency_ms=it_data.get("total_latency_ms", 0.0),
                prompt_tokens=it_data.get("prompt_tokens", 0),
                completion_tokens=it_data.get("completion_tokens", 0),
                error=it_data.get("error"),
            ))
        result.scenarios[name] = ss

    return result

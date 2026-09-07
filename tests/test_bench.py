"""Tests for ppmlx.bench — benchmark runner, aggregation, and I/O."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from ppmlx.bench import (
    BenchmarkRunner,
    BenchmarkResult,
    IterationResult,
    ScenarioStats,
    SCENARIOS,
    print_results,
    print_comparison,
    save_results,
    load_results,
)


# ── IterationResult / ScenarioStats ─────────────────────────────────


def _make_iterations(values: list[tuple[float, float, float, int, int]]) -> list[IterationResult]:
    """Helper: create IterationResult list from (ttft, tps, latency, prompt, compl) tuples."""
    return [
        IterationResult(
            ttft_ms=v[0],
            tokens_per_sec=v[1],
            total_latency_ms=v[2],
            prompt_tokens=v[3],
            completion_tokens=v[4],
        )
        for v in values
    ]


class TestScenarioStats:
    def test_stats_basic(self):
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = _make_iterations([
            (100.0, 50.0, 2000.0, 10, 100),
            (120.0, 60.0, 1800.0, 10, 120),
            (110.0, 55.0, 1900.0, 10, 110),
        ])
        stats = ss.stats()

        # avg ttft = (100+120+110)/3 = 110
        assert stats["ttft_ms"]["avg"] == 110.0
        assert stats["ttft_ms"]["min"] == 100.0
        assert stats["ttft_ms"]["max"] == 120.0

        # avg tps = (50+60+55)/3 = 55
        assert stats["tokens_per_sec"]["avg"] == 55.0

        # Sample standard deviation uses n - 1: sqrt(50 / 2) = 5.
        assert stats["tokens_per_sec"]["stddev"] == 5.0

    def test_stats_empty(self):
        ss = ScenarioStats(scenario="simple", label="Simple")
        stats = ss.stats()
        assert stats["ttft_ms"]["avg"] == 0.0
        assert stats["tokens_per_sec"]["avg"] == 0.0

    def test_stats_with_errors(self):
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = [
            IterationResult(ttft_ms=100.0, tokens_per_sec=50.0, total_latency_ms=2000.0,
                            prompt_tokens=10, completion_tokens=100),
            IterationResult(error="connection refused"),
            IterationResult(ttft_ms=120.0, tokens_per_sec=60.0, total_latency_ms=1800.0,
                            prompt_tokens=10, completion_tokens=120),
        ]
        stats = ss.stats()
        # Only 2 successful iterations
        assert len(ss.successful) == 2
        assert stats["ttft_ms"]["avg"] == 110.0

    def test_successful_property(self):
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = [
            IterationResult(ttft_ms=100.0, tokens_per_sec=50.0, total_latency_ms=2000.0,
                            prompt_tokens=10, completion_tokens=100),
            IterationResult(error="fail"),
        ]
        assert len(ss.successful) == 1
        assert ss.successful[0].ttft_ms == 100.0

    def test_single_iteration_stddev_zero(self):
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = _make_iterations([(100.0, 50.0, 2000.0, 10, 100)])
        stats = ss.stats()
        assert stats["ttft_ms"]["stddev"] == 0.0
        assert stats["tokens_per_sec"]["stddev"] == 0.0


# ── BenchmarkResult serialization ────────────────────────────────────


class TestBenchmarkResult:
    def _make_result(self) -> BenchmarkResult:
        result = BenchmarkResult(
            model="test-model",
            timestamp="2026-01-01T00:00:00Z",
            runs=2,
            system_info={"platform": "test", "ram_gb": 32.0},
        )
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = _make_iterations([
            (100.0, 50.0, 2000.0, 10, 100),
            (120.0, 60.0, 1800.0, 10, 120),
        ])
        result.scenarios["simple"] = ss
        return result

    def test_to_dict(self):
        result = self._make_result()
        d = result.to_dict()
        assert d["model"] == "test-model"
        assert d["runs"] == 2
        assert "simple" in d["results"]
        assert "stats" in d["results"]["simple"]
        assert "iterations" in d["results"]["simple"]
        assert len(d["results"]["simple"]["iterations"]) == 2

    def test_to_dict_iteration_fields(self):
        result = self._make_result()
        d = result.to_dict()
        it = d["results"]["simple"]["iterations"][0]
        assert it["ttft_ms"] == 100.0
        assert it["tokens_per_sec"] == 50.0
        assert it["total_latency_ms"] == 2000.0
        assert it["prompt_tokens"] == 10
        assert it["completion_tokens"] == 100
        assert it["error"] is None


# ── JSON save/load round-trip ────────────────────────────────────────


class TestJsonIO:
    def test_save_and_load(self, tmp_path: Path):
        result = BenchmarkResult(
            model="llama3",
            timestamp="2026-01-01T00:00:00Z",
            runs=3,
            system_info={"platform": "test"},
        )
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = _make_iterations([
            (100.0, 50.0, 2000.0, 10, 100),
            (120.0, 60.0, 1800.0, 10, 120),
            (110.0, 55.0, 1900.0, 10, 110),
        ])
        result.scenarios["simple"] = ss

        out_path = tmp_path / "results.json"
        save_results(result, out_path)
        assert out_path.exists()

        loaded = load_results(out_path)
        assert loaded.model == "llama3"
        assert loaded.runs == 3
        assert "simple" in loaded.scenarios
        assert len(loaded.scenarios["simple"].iterations) == 3
        assert loaded.scenarios["simple"].iterations[0].ttft_ms == 100.0

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        result = BenchmarkResult(
            model="llama3",
            timestamp="2026-01-01T00:00:00Z",
            runs=1,
            system_info={},
        )
        nested = tmp_path / "a" / "b" / "results.json"
        save_results(result, nested)
        assert nested.exists()

    def test_load_preserves_errors(self, tmp_path: Path):
        result = BenchmarkResult(
            model="test",
            timestamp="2026-01-01T00:00:00Z",
            runs=2,
            system_info={},
        )
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = [
            IterationResult(ttft_ms=100.0, tokens_per_sec=50.0, total_latency_ms=2000.0,
                            prompt_tokens=10, completion_tokens=100),
            IterationResult(error="connection refused"),
        ]
        result.scenarios["simple"] = ss

        path = tmp_path / "results.json"
        save_results(result, path)
        loaded = load_results(path)
        assert loaded.scenarios["simple"].iterations[0].error is None
        assert loaded.scenarios["simple"].iterations[1].error == "connection refused"

    def test_json_format_valid(self, tmp_path: Path):
        result = BenchmarkResult(
            model="test",
            timestamp="2026-01-01T00:00:00Z",
            runs=1,
            system_info={"ram_gb": 32},
        )
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = _make_iterations([(100.0, 50.0, 2000.0, 10, 100)])
        result.scenarios["simple"] = ss

        path = tmp_path / "results.json"
        save_results(result, path)

        # Verify raw JSON structure
        with open(path) as f:
            raw = json.load(f)

        assert raw["model"] == "test"
        assert raw["system_info"]["ram_gb"] == 32
        assert "results" in raw
        assert "simple" in raw["results"]
        assert "stats" in raw["results"]["simple"]
        stats = raw["results"]["simple"]["stats"]
        for key in ["ttft_ms", "tokens_per_sec", "total_latency_ms", "prompt_tokens", "completion_tokens"]:
            assert key in stats
            for stat_key in ["avg", "stddev", "min", "max"]:
                assert stat_key in stats[key]


# ── BenchmarkRunner ──────────────────────────────────────────────────


class TestBenchmarkRunner:
    def test_invalid_scenario(self):
        with pytest.raises(ValueError, match="Unknown scenarios"):
            BenchmarkRunner(model="test", scenarios=["nonexistent"])

    def test_default_scenarios(self):
        runner = BenchmarkRunner(model="test")
        assert runner.scenario_names == list(SCENARIOS.keys())

    def test_custom_scenarios(self):
        runner = BenchmarkRunner(model="test", scenarios=["simple", "complex"])
        assert runner.scenario_names == ["simple", "complex"]

    def test_run_with_mocked_http(self):
        """Test full run with mocked HTTP streaming responses."""
        def make_sse_body(n_tokens: int) -> str:
            lines = []
            for i in range(n_tokens):
                chunk = {
                    "choices": [{"delta": {"content": f"word{i} "}}],
                }
                lines.append(f"data: {json.dumps(chunk)}\n\n")
            lines.append("data: [DONE]\n\n")
            return "".join(lines)

        def make_mock_response():
            """Create a fresh mock response for each stream() call."""
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.iter_text = MagicMock(return_value=iter([make_sse_body(10)]))
            resp.__enter__ = MagicMock(return_value=resp)
            resp.__exit__ = MagicMock(return_value=False)
            return resp

        mock_client = MagicMock()
        mock_client.stream = MagicMock(side_effect=lambda *a, **kw: make_mock_response())
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch("ppmlx.bench.httpx.Client", return_value=mock_client):
            runner = BenchmarkRunner(model="test", runs=2, scenarios=["simple"])
            result = runner.run()

        assert result.model == "test"
        assert result.runs == 2
        assert "simple" in result.scenarios
        ss = result.scenarios["simple"]
        assert len(ss.iterations) == 2
        # Each iteration should have found 10 tokens
        for it in ss.iterations:
            assert it.error is None
            assert it.completion_tokens == 10
            assert it.total_latency_ms > 0

    def test_run_connection_error(self):
        """Test that connection errors are captured per-iteration, not raised."""
        import httpx as _httpx

        with patch("ppmlx.bench.httpx.Client") as MockClient:
            mock_client = MagicMock()
            mock_client.__enter__ = MagicMock(return_value=mock_client)
            mock_client.__exit__ = MagicMock(return_value=False)
            mock_client.stream = MagicMock(
                side_effect=_httpx.ConnectError("Connection refused")
            )
            MockClient.return_value = mock_client

            runner = BenchmarkRunner(model="test", runs=1, scenarios=["simple"])
            result = runner.run()

        ss = result.scenarios["simple"]
        assert len(ss.iterations) == 1
        assert ss.iterations[0].error is not None
        assert "Connection refused" in ss.iterations[0].error


# ── Comparison logic ─────────────────────────────────────────────────


class TestComparison:
    def _make_result(self, ttft: float, tps: float, latency: float) -> BenchmarkResult:
        result = BenchmarkResult(
            model="test",
            timestamp="2026-01-01T00:00:00Z",
            runs=1,
            system_info={},
        )
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = _make_iterations([(ttft, tps, latency, 10, 100)])
        result.scenarios["simple"] = ss
        return result

    def test_comparison_delta_computed(self):
        baseline = self._make_result(100.0, 50.0, 2000.0)
        current = self._make_result(90.0, 60.0, 1800.0)

        # TTFT went from 100 to 90 → -10%
        cur_stats = current.scenarios["simple"].stats()
        base_stats = baseline.scenarios["simple"].stats()

        ttft_delta = ((cur_stats["ttft_ms"]["avg"] - base_stats["ttft_ms"]["avg"])
                      / base_stats["ttft_ms"]["avg"]) * 100
        assert abs(ttft_delta - (-10.0)) < 0.01

        # TPS went from 50 to 60 → +20%
        tps_delta = ((cur_stats["tokens_per_sec"]["avg"] - base_stats["tokens_per_sec"]["avg"])
                     / base_stats["tokens_per_sec"]["avg"]) * 100
        assert abs(tps_delta - 20.0) < 0.01

    def test_print_comparison_no_crash(self, capsys):
        """Ensure print_comparison runs without error."""
        from rich.console import Console

        baseline = self._make_result(100.0, 50.0, 2000.0)
        current = self._make_result(90.0, 60.0, 1800.0)
        c = Console(file=io.StringIO())
        print_comparison(current, baseline, c)

    def test_comparison_missing_scenario(self):
        """Comparison skips scenarios not in baseline."""
        baseline = BenchmarkResult(
            model="test", timestamp="", runs=1, system_info={},
        )
        current = BenchmarkResult(
            model="test", timestamp="", runs=1, system_info={},
        )
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = _make_iterations([(100.0, 50.0, 2000.0, 10, 100)])
        current.scenarios["simple"] = ss
        # baseline has no scenarios — comparison should not crash
        from rich.console import Console
        c = Console(file=io.StringIO())
        print_comparison(current, baseline, c)


# ── Display ──────────────────────────────────────────────────────────


class TestDisplay:
    def test_print_results_no_crash(self, capsys):
        result = BenchmarkResult(
            model="test",
            timestamp="2026-01-01T00:00:00Z",
            runs=1,
            system_info={"platform": "test", "ram_gb": 32},
        )
        ss = ScenarioStats(scenario="simple", label="Simple")
        ss.iterations = _make_iterations([(100.0, 50.0, 2000.0, 10, 100)])
        result.scenarios["simple"] = ss

        from rich.console import Console
        c = Console(file=io.StringIO())
        print_results(result, c)

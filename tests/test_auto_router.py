"""Tests for ppmlx.auto_router — smart model routing."""
from __future__ import annotations

import pytest

from ppmlx.auto_router import (
    RouterConfig,
    RouteDecision,
    analyze_complexity,
    route,
)


# ── Complexity analysis ─────────────────────────────────────────────────

class TestAnalyzeComplexity:
    def test_simple_greeting(self):
        msgs = [{"role": "user", "content": "Hello"}]
        score = analyze_complexity(msgs)
        assert score <= 2

    def test_simple_question(self):
        msgs = [{"role": "user", "content": "What is 2+2?"}]
        score = analyze_complexity(msgs)
        assert score <= 2

    def test_short_factual(self):
        msgs = [{"role": "user", "content": "List the planets in our solar system"}]
        score = analyze_complexity(msgs)
        assert score <= 3

    def test_code_generation(self):
        msgs = [{"role": "user", "content": (
            "Write a Python function that implements binary search. "
            "Include error handling and type hints. "
            "```python\ndef binary_search(arr, target):\n    pass\n```"
        )}]
        score = analyze_complexity(msgs)
        assert score >= 3

    def test_architecture_question(self):
        msgs = [{"role": "user", "content": (
            "Design a distributed rate limiter for an API gateway. "
            "Analyze the trade-offs between token bucket vs sliding window. "
            "Explain why one might be better for bursty traffic. "
            "Compare the performance characteristics and provide pseudocode."
        )}]
        score = analyze_complexity(msgs)
        assert score >= 3  # reasoning keywords dominate for short-but-complex prompts

    def test_long_prompt_is_complex(self):
        long_text = "x " * 2000  # ~2000 tokens
        msgs = [{"role": "user", "content": long_text}]
        score = analyze_complexity(msgs)
        assert score >= 3

    def test_multi_turn_adds_complexity(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is..."},
            {"role": "user", "content": "What about async?"},
            {"role": "assistant", "content": "Async in Python..."},
            {"role": "user", "content": "Can you compare it to Go?"},
            {"role": "assistant", "content": "Go vs Python..."},
            {"role": "user", "content": "Which should I use?"},
        ]
        score = analyze_complexity(msgs)
        assert score >= 3

    def test_tools_add_complexity(self):
        msgs = [{"role": "user", "content": "Search for files"}]
        tools = [
            {"type": "function", "function": {"name": "bash", "description": "Run bash"}},
            {"type": "function", "function": {"name": "read", "description": "Read file"}},
        ]
        score_no_tools = analyze_complexity(msgs)
        score_with_tools = analyze_complexity(msgs, tools=tools)
        assert score_with_tools > score_no_tools

    def test_many_tools_extra_complex(self):
        msgs = [{"role": "user", "content": "Help me"}]
        tools = [
            {"type": "function", "function": {"name": f"tool_{i}", "description": f"Tool {i}"}}
            for i in range(8)
        ]
        score = analyze_complexity(msgs, tools=tools)
        assert score >= 3

    def test_high_max_tokens(self):
        msgs = [{"role": "user", "content": "Write a story"}]
        score_low = analyze_complexity(msgs, max_tokens=256)
        score_high = analyze_complexity(msgs, max_tokens=8192)
        assert score_high > score_low

    def test_image_content_is_complex(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]}]
        score = analyze_complexity(msgs)
        assert score >= 3

    def test_score_capped_at_10(self):
        # Throw everything at it
        long_code = "```python\n" + "\n".join(
            f"def func_{i}(): pass  # analyze and optimize" for i in range(100)
        ) + "\n```"
        msgs = [
            {"role": "system", "content": "You are an expert architect."},
            *[{"role": "user", "content": f"Turn {i}: explain and debug"} for i in range(10)],
            {"role": "user", "content": long_code},
        ]
        tools = [{"type": "function", "function": {"name": f"t{i}", "description": ""}} for i in range(10)]
        score = analyze_complexity(msgs, tools=tools, max_tokens=16384)
        assert score == 10


# ── Routing decisions ───────────────────────────────────────────────────

class TestRoute:
    @pytest.fixture
    def config(self):
        return RouterConfig(
            enabled=True,
            small_model="small/model",
            large_model="large/model",
            threshold=3,
        )

    def test_simple_routes_to_small(self, config):
        msgs = [{"role": "user", "content": "Hello"}]
        decision = route(msgs, config)
        assert decision.model == "small/model"
        assert decision.complexity_score < 3

    def test_complex_routes_to_large(self, config):
        msgs = [{"role": "user", "content": (
            "Design a microservices architecture for a real-time trading platform. "
            "Analyze the trade-offs between event sourcing and CQRS. "
            "```python\nclass OrderBook:\n    def __init__(self):\n        self.bids = {}\n```\n"
            "Explain the pros and cons of each approach step by step."
        )}]
        decision = route(msgs, config)
        assert decision.model == "large/model"
        assert decision.complexity_score >= 3

    def test_threshold_boundary(self, config):
        # Score exactly at threshold should use large
        msgs = [{"role": "user", "content": "Explain the difference between TCP and UDP in detail"}]
        decision = route(msgs, config)
        # Just verify the decision is consistent with the score vs threshold
        if decision.complexity_score >= config.threshold:
            assert decision.model == "large/model"
        else:
            assert decision.model == "small/model"

    def test_high_threshold_routes_most_to_small(self):
        config = RouterConfig(
            enabled=True,
            small_model="fast/model",
            large_model="big/model",
            threshold=8,  # Very high — only extreme complexity uses big
        )
        msgs = [{"role": "user", "content": "Write a quick function to reverse a string"}]
        decision = route(msgs, config)
        assert decision.model == "fast/model"

    def test_low_threshold_routes_most_to_large(self):
        config = RouterConfig(
            enabled=True,
            small_model="fast/model",
            large_model="big/model",
            threshold=1,  # Very low — everything uses big
        )
        msgs = [{"role": "user", "content": "hi"}]
        decision = route(msgs, config)
        assert decision.model == "big/model"

    def test_decision_includes_reason(self, config):
        msgs = [{"role": "user", "content": "Hello"}]
        decision = route(msgs, config)
        assert "threshold" in decision.reason
        assert str(config.threshold) in decision.reason

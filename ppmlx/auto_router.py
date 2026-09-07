"""Smart model router — automatically selects the best model for each request.

When the client sends ``model: "auto"`` (or any alias mapped to the router),
ppmlx analyzes the request and routes it to the optimal model:

- **Simple** requests (short prompt, no tools, no images) → small fast model
- **Complex** requests (long prompt, tools, multi-turn, code) → large capable model

Configuration via ``~/.ppmlx/config.toml``:

.. code-block:: toml

    [router]
    enabled = true
    small_model = "qwen3.5:0.8b"
    large_model = "qwen3.5:9b"
    threshold = 3                     # complexity score threshold (1-10)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from ppmlx.config import RouterConfig

log = logging.getLogger("ppmlx.auto_router")


@dataclass(frozen=True)
class RouteDecision:
    """Result of routing analysis."""
    model: str
    complexity_score: int
    reason: str


# ── Complexity signals ──────────────────────────────────────────────────

# Match unambiguous code constructs only (avoid natural-language false positives)
_CODE_PATTERNS = re.compile(
    r"```|def\s+\w+\(|class\s+\w+[:(]|#include\s*<|"
    r"SELECT\s+\w+\s+FROM|CREATE\s+TABLE|async\s+def\s",
    re.IGNORECASE,
)

_REASONING_KEYWORDS = re.compile(
    r"\bexplain\b|\banalyze\b|\bcompare\b|\bdesign\b|\barchitect\b|"
    r"\bdebug\b|\boptimize\b|\brefactor\b|\breview\b|"
    r"\btrade.?off\b|\bpros?\s+and\s+cons\b|\bstep.by.step\b",
    re.IGNORECASE,
)

_SIMPLE_PATTERNS = re.compile(
    r"(yes|no|ok|sure|thanks|hello|hi|hey|what is|list|show|get|"
    r"tell me|name|how many|true or false)\b",
    re.IGNORECASE,
)


def _count_tokens_approx(text: str) -> int:
    """Rough token count (~4 chars per token for Latin text)."""
    return max(1, len(text) // 4)


def _extract_text(messages: list[dict]) -> str:
    """Extract all text content from messages."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(part.get("text", ""))
    return "\n".join(parts)


def analyze_complexity(
    messages: list[dict],
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
) -> int:
    """Score request complexity from 1 (trivial) to 10 (very complex)."""
    score = 1
    text = _extract_text(messages)
    approx_tokens = _count_tokens_approx(text)

    # Message count (multi-turn = more complex)
    n_msgs = len(messages)
    if n_msgs >= 10:
        score += 2
    elif n_msgs >= 4:
        score += 1

    # Prompt length
    if approx_tokens > 2000:
        score += 3
    elif approx_tokens > 500:
        score += 2
    elif approx_tokens > 100:
        score += 1

    # Code presence
    code_matches = len(_CODE_PATTERNS.findall(text))
    if code_matches >= 3:
        score += 2
    elif code_matches >= 1:
        score += 1

    # Reasoning keywords
    reasoning_matches = len(_REASONING_KEYWORDS.findall(text))
    if reasoning_matches >= 3:
        score += 2
    elif reasoning_matches >= 1:
        score += 1

    # Tools
    if tools:
        score += 1
        if len(tools) > 5:
            score += 1

    # Images — add +2 once if any message contains an image
    has_images = any(
        isinstance(msg.get("content"), list)
        and any(
            isinstance(p, dict) and p.get("type") == "image_url"
            for p in msg["content"]
        )
        for msg in messages
    )
    if has_images:
        score += 2

    # Requested output length
    if max_tokens and max_tokens > 4096:
        score += 1

    # Simplicity detection — last user message starts with a simple pattern
    last_user = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            c = msg.get("content", "")
            if isinstance(c, str):
                last_user = c
            elif isinstance(c, list):
                last_user = " ".join(
                    p.get("text", "") for p in c
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            break
    if last_user and _SIMPLE_PATTERNS.match(last_user.strip()) and approx_tokens < 100:
        score = max(1, score - 2)

    return min(10, score)


def route(
    messages: list[dict],
    config: RouterConfig,
    tools: list[dict] | None = None,
    max_tokens: int | None = None,
) -> RouteDecision:
    """Decide which model to use based on request complexity."""
    score = analyze_complexity(messages, tools=tools, max_tokens=max_tokens)

    if score >= config.threshold:
        return RouteDecision(
            model=config.large_model,
            complexity_score=score,
            reason=f"complexity={score} >= threshold={config.threshold} -> large model",
        )
    return RouteDecision(
        model=config.small_model,
        complexity_score=score,
        reason=f"complexity={score} < threshold={config.threshold} -> small model",
    )

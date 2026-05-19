"""Long-session evals for rolling-context compaction.

The compact eval complements memory_eval: it checks whether a long task session can
be reduced to a hot tail + rendered graph context without losing critical state.
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ppmlx.context_reducer import ContextBudget, ContextReducer, group_messages_into_episodes
from ppmlx.memory_engine import MemoryEngine
from ppmlx.memory_store import MemoryStore


@dataclass
class CompactEvalCase:
    id: str
    description: str
    project_id: str
    session_id: str
    messages: list[dict[str, Any]]
    expected_terms: list[str]
    forbidden_terms: list[str] = field(default_factory=list)
    expected_session_context_terms: list[str] = field(default_factory=list)
    max_reduced_tokens: int = 10_000
    min_compression_ratio: float = 5.0
    min_continuity_score: float = 0.9
    min_session_context_coverage: float = 0.85


@dataclass
class CompactCaseResult:
    case_id: str
    passed: bool
    original_tokens: int
    reduced_tokens: int
    compression_ratio: float
    context_items: int
    cold_messages: int
    expected_terms: list[str]
    found_terms: list[str]
    missed_terms: list[str]
    forbidden_terms: list[str]
    wrong_terms: list[str]
    continuity_score: float
    session_context_coverage: float
    session_context_found_terms: list[str]
    session_context_missed_terms: list[str]
    session_context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "passed": self.passed,
            "original_tokens": self.original_tokens,
            "reduced_tokens": self.reduced_tokens,
            "compression_ratio": self.compression_ratio,
            "context_items": self.context_items,
            "cold_messages": self.cold_messages,
            "expected_terms": self.expected_terms,
            "found_terms": self.found_terms,
            "missed_terms": self.missed_terms,
            "forbidden_terms": self.forbidden_terms,
            "wrong_terms": self.wrong_terms,
            "continuity_score": self.continuity_score,
            "session_context_coverage": self.session_context_coverage,
            "session_context_found_terms": self.session_context_found_terms,
            "session_context_missed_terms": self.session_context_missed_terms,
            "session_context": self.session_context,
        }


@dataclass
class CompactEvalReport:
    timestamp: str
    passed: bool
    summary: dict[str, Any]
    cases: list[CompactCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "summary": self.summary,
            "cases": [case.to_dict() for case in self.cases],
        }


class CompactEvalRunner:
    def __init__(self, budget: ContextBudget | None = None):
        self.budget = budget or ContextBudget(
            mode="compact",
            compact_threshold_tokens=1_500,
            hot_tail_tokens=900,
            session_context_tokens=2_000,
            max_context_items=40,
        )

    def run(self, cases: list[CompactEvalCase] | None = None) -> CompactEvalReport:
        cases = cases or builtin_cases()
        results: list[CompactCaseResult] = []
        for case in cases:
            results.append(self._run_case(case))
        passed = all(result.passed for result in results)
        summary = {
            "cases": len(results),
            "passed_cases": sum(1 for result in results if result.passed),
            "avg_compression_ratio": round(
                sum(result.compression_ratio for result in results) / max(len(results), 1), 2
            ),
            "avg_continuity_score": round(
                sum(result.continuity_score for result in results) / max(len(results), 1), 4
            ),
            "missed_terms": sum(len(result.missed_terms) for result in results),
            "context_missed_terms": sum(len(result.session_context_missed_terms) for result in results),
            "wrong_terms": sum(len(result.wrong_terms) for result in results),
            "avg_session_context_coverage": round(
                sum(result.session_context_coverage for result in results) / max(len(results), 1), 4
            ),
            "max_reduced_tokens": max((result.reduced_tokens for result in results), default=0),
        }
        return CompactEvalReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            passed=passed,
            summary=summary,
            cases=results,
        )

    def _run_case(self, case: CompactEvalCase) -> CompactCaseResult:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.db")
            store.init()
            engine = MemoryEngine(store=store)
            reducer = ContextReducer(self.budget, store=store, engine=engine)
            self._preingest_case(case, reducer=reducer, engine=engine)
            result = reducer.reduce(
                request_id=f"compact-eval-{case.id}",
                model_alias="compact-eval-model",
                model_repo="compact-eval/model",
                messages=case.messages,
                memory_context={
                    "project_id": case.project_id,
                    "session_id": case.session_id,
                    "metadata": {"eval_case": case.id},
                },
            )

        session_context = _extract_session_context(result.messages)
        haystack = _normalize("\n".join(_message_text(message) for message in result.messages))
        found = [term for term in case.expected_terms if _normalize(term) in haystack]
        missed = [term for term in case.expected_terms if term not in found]
        wrong = [term for term in case.forbidden_terms if _normalize(term) in haystack]
        context_terms = case.expected_session_context_terms or case.expected_terms
        normalized_session_context = _normalize(session_context)
        context_found = [term for term in context_terms if _normalize(term) in normalized_session_context]
        context_missed = [term for term in context_terms if term not in context_found]
        continuity_score = round(len(found) / max(len(case.expected_terms), 1), 4)
        session_context_coverage = round(len(context_found) / max(len(context_terms), 1), 4)
        compression_ratio = round(result.original_tokens / max(result.reduced_tokens, 1), 2)
        passed = (
            result.reduced_tokens <= case.max_reduced_tokens
            and compression_ratio >= case.min_compression_ratio
            and continuity_score >= case.min_continuity_score
            and session_context_coverage >= case.min_session_context_coverage
            and not wrong
        )
        return CompactCaseResult(
            case_id=case.id,
            passed=passed,
            original_tokens=result.original_tokens,
            reduced_tokens=result.reduced_tokens,
            compression_ratio=compression_ratio,
            context_items=result.context_items,
            cold_messages=result.cold_messages,
            expected_terms=case.expected_terms,
            found_terms=found,
            missed_terms=missed,
            forbidden_terms=case.forbidden_terms,
            wrong_terms=wrong,
            continuity_score=continuity_score,
            session_context_coverage=session_context_coverage,
            session_context_found_terms=context_found,
            session_context_missed_terms=context_missed,
            session_context=session_context,
        )

    @staticmethod
    def _preingest_case(case: CompactEvalCase, *, reducer: ContextReducer, engine: MemoryEngine) -> None:
        _, cold = reducer._select_hot_tail([message for message in case.messages if message.get("role") != "system"])  # noqa: SLF001
        for episode in group_messages_into_episodes(cold):
            if not episode.messages:
                continue
            engine.capture_chat(
                request_id=f"compact-eval-{case.id}-preingest-e{episode.index}",
                endpoint="/v1/chat/completions#preingest",
                model_alias="compact-eval-model",
                model_repo="compact-eval/model",
                messages=episode.messages,
                response_text=None,
                project_id=case.project_id,
                session_id=case.session_id,
                metadata={"eval_case": case.id, "preingest": True},
            )


def builtin_cases() -> list[CompactEvalCase]:
    return [tv_buying_case(), tv_buying_json_tool_trace_case(), real_project_handoff_case()]


def tv_buying_case() -> CompactEvalCase:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You help with careful product research. Keep state across long sessions."},
        {"role": "user", "content": "I am buying a TV. Budget <= 5000 PLN. Screen size 55-65 inch. Need HDMI 2.1 for PS5."},
        {"role": "assistant", "content": "I will track constraints and compare current Polish offers."},
        {"role": "user", "content": "I prefer OLED if burn-in risk is acceptable. I prefer concise comparison tables."},
        {"role": "assistant", "content": "Got it: OLED preferred, concise tables."},
    ]

    # Synthetic long tool/MCP churn.  These messages should be compacted away as
    # raw text but distilled facts should remain in the graph.
    for batch in range(28):
        messages.extend([
            {"role": "user", "content": f"Search batch {batch}: find TV offers and specs."},
            {"role": "assistant", "content": "Calling product_search, price_check, review_lookup."},
            {
                "role": "tool",
                "name": "product_search",
                "content": _tool_payload(batch),
            },
            {
                "role": "assistant",
                "content": _episode_note(batch),
            },
        ])

    messages.extend([
        {"role": "user", "content": "Before final recommendation, what are the best two options and what is still unknown?"},
    ])

    return CompactEvalCase(
        id="tv_buying_long_session",
        description="Long TV shopping session with many tool results should compact to key state.",
        project_id="tv-shopping",
        session_id="tv-session-001",
        messages=messages,
        expected_terms=[
            "budget = 5000 PLN",
            "screen_size = 55-65 inch",
            "requires = HDMI 2.1 for PS5",
            "User prefers OLED if burn-in risk is acceptable",
            "Current shortlist: LG OLED C4, Samsung S90D",
            "Rejected Samsung CU8000: 60Hz and no HDMI 2.1",
            "tv-shopping todo: ask room brightness and viewing distance",
        ],
        forbidden_terms=[
            "Samsung CU8000 is recommended",
            "budget = 8000 PLN",
        ],
        max_reduced_tokens=10_000,
        min_compression_ratio=4.0,
        min_continuity_score=0.85,
    )


def tv_buying_json_tool_trace_case() -> CompactEvalCase:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You help with careful product research. Keep state across long sessions."},
        {"role": "user", "content": "I am buying a TV. Budget <= 5000 PLN. Need HDMI 2.1 for PS5."},
        {"role": "assistant", "content": "I will search structured product data."},
    ]
    for batch in range(18):
        messages.extend([
            {"role": "user", "content": f"Structured tool search batch {batch}."},
            {"role": "assistant", "content": "Calling product_search_json."},
            {
                "role": "tool",
                "name": "product_search_json",
                "content": json.dumps(_json_tool_payload(batch)),
            },
            {"role": "assistant", "content": "Tool results received; continue."},
        ])
    messages.append({"role": "user", "content": "Use the structured search results. Which candidate is strongest and what was rejected?"})

    return CompactEvalCase(
        id="tv_buying_json_tool_trace",
        description="Structured JSON tool outputs should distill into product facts without raw JSON context.",
        project_id="tv-shopping-json",
        session_id="tv-session-json-001",
        messages=messages,
        expected_terms=[
            "budget = 5000 PLN",
            "requires = HDMI 2.1 for PS5",
            "Candidate: LG OLED C4",
            "LG OLED C4 price: 4599 PLN",
            "LG OLED C4 spec panel = OLED",
            "LG OLED C4 spec hdmi_2_1 = 4",
            "Rejected Samsung CU8000: 60Hz and no HDMI 2.1",
        ],
        forbidden_terms=[
            "api_key",
            "Samsung CU8000 is recommended",
            "budget = 8000 PLN",
        ],
        max_reduced_tokens=10_000,
        min_compression_ratio=4.0,
        min_continuity_score=0.85,
    )


def real_project_handoff_case() -> CompactEvalCase:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "You are a coding assistant. Preserve project state across long sessions."},
        {
            "role": "user",
            "content": (
                "Goal: improve ppmlx synthetic memory evals so they reflect real-session quality. "
                "Need exact benchmark identifiers in reports."
            ),
        },
        {"role": "assistant", "content": "I will track the eval work and avoid hiding quality gaps."},
        {
            "role": "user",
            "content": (
                "Decision: real-session quality-bench failures should drive synthetic benchmark design. "
                "Rejected synthetic-only PASS as sufficient because real sessions showed low recall."
            ),
        },
        {"role": "assistant", "content": "Recorded the decision and rejection rationale."},
    ]

    # Real sessions contain lots of unrelated tool output and old tasks. The
    # eval should prove that project handoff facts survive graph compaction while
    # unrelated fixture facts do not dominate the rendered context.
    for batch in range(16):
        messages.extend([
            {"role": "user", "content": f"Investigate benchmark batch {batch}."},
            {
                "role": "tool",
                "name": "bash",
                "content": "\n".join(
                    f"noise_{batch}_{idx}: unrelated fixture output with travel budget 800 PLN and app_{idx}.py"
                    for idx in range(45)
                ),
            },
            {"role": "assistant", "content": _project_handoff_note(batch)},
        ])

    messages.append({"role": "user", "content": "Give the current ppmlx memory benchmark handoff and next action."})

    return CompactEvalCase(
        id="ppmlx_real_project_handoff",
        description="Project handoff facts should survive realistic noisy session compaction.",
        project_id="ppmlx",
        session_id="ppmlx-bench-session-001",
        messages=messages,
        expected_terms=[
            "Goal: improve ppmlx synthetic memory evals so they reflect real-session quality",
            "Need exact benchmark identifiers in reports",
            "Decision: real-session quality-bench failures should drive synthetic benchmark design",
            "Rejected synthetic-only PASS as sufficient because real sessions showed low recall",
            "ppmlx todo: rerun answerable real-session batch with include-content",
            "ppmlx todo: add context coverage metrics to compact-eval",
        ],
        expected_session_context_terms=[
            "Goal: improve ppmlx synthetic memory evals so they reflect real-session quality",
            "Need exact benchmark identifiers in reports",
            "Decision: real-session quality-bench failures should drive synthetic benchmark design",
            "Rejected synthetic-only PASS as sufficient because real sessions showed low recall",
            "ppmlx todo: rerun answerable real-session batch with include-content",
            "ppmlx todo: add context coverage metrics to compact-eval",
        ],
        forbidden_terms=[
            "budget = 800 PLN",
            "travel budget 800 PLN",
            "synthetic-only PASS is sufficient",
        ],
        max_reduced_tokens=10_000,
        min_compression_ratio=4.0,
        min_continuity_score=0.85,
        min_session_context_coverage=0.85,
    )


def save_report(report: CompactEvalReport, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    return out


def _tool_payload(batch: int) -> str:
    filler = "\n".join(
        f"offer_{batch}_{i}: random LED TV, {3200 + i} PLN, noisy description, repeated spec blob"
        for i in range(80)
    )
    if batch == 3:
        return filler + "\nRemember that LG OLED C4 costs 4599 PLN and matches HDMI 2.1."
    if batch == 8:
        return filler + "\nRemember that Samsung S90D costs 4999 PLN and has better brightness."
    if batch == 12:
        return filler + "\nRejected Samsung CU8000 because 60Hz and no HDMI 2.1."
    if batch == 16:
        return filler + "\nShortlist: LG OLED C4, Samsung S90D."
    if batch == 20:
        return filler + "\nTodo: ask room brightness and viewing distance."
    return filler


def _json_tool_payload(batch: int) -> dict[str, Any]:
    products: list[dict[str, Any]] = [
        {
            "name": f"Generic LED {batch}",
            "price": {"amount": 3200 + batch, "currency": "PLN"},
            "availability": "unknown",
            "specs": {"panel": "LED", "refresh_rate": "60Hz"},
        }
        for _ in range(12)
    ]
    if batch == 4:
        products.append({
            "name": "LG OLED C4",
            "price": {"amount": 4599, "currency": "PLN"},
            "availability": "in_stock",
            "url": "https://example.test/lg-oled-c4",
            "specs": {"panel": "OLED", "hdmi_2_1": 4, "refresh_rate": "120Hz"},
        })
    if batch == 9:
        products.append({
            "name": "Samsung CU8000",
            "price": {"amount": 2999, "currency": "PLN"},
            "rejected": True,
            "reason": "60Hz and no HDMI 2.1",
            "specs": {"panel": "LED", "refresh_rate": "60Hz"},
        })
    return {"products": products, "batch": batch, "tool": "product_search_json"}


def _episode_note(batch: int) -> str:
    if batch == 0:
        return "Decision: budget = 5000 PLN. Decision: screen_size = 55-65 inch. Decision: requires = HDMI 2.1 for PS5."
    if batch == 3:
        return "Remember that LG OLED C4 is a strong candidate at 4599 PLN."
    if batch == 8:
        return "Remember that Samsung S90D is a strong candidate at 4999 PLN."
    if batch == 12:
        return "Rejected Samsung CU8000 because 60Hz and no HDMI 2.1."
    if batch == 16:
        return "Shortlist: LG OLED C4, Samsung S90D."
    if batch == 20:
        return "Todo: ask room brightness and viewing distance."
    return "No durable decision from this batch. Continue search."


def _project_handoff_note(batch: int) -> str:
    if batch == 1:
        return "Goal: improve ppmlx synthetic memory evals so they reflect real-session quality."
    if batch == 3:
        return "Remember that ppmlx reports need exact benchmark identifiers."
    if batch == 5:
        return "Decision: real-session quality-bench failures should drive synthetic benchmark design."
    if batch == 7:
        return "Rejected synthetic-only PASS because real sessions showed low recall."
    if batch == 10:
        return "Todo: rerun answerable real-session batch with include-content."
    if batch == 12:
        return "Todo: add context coverage metrics to compact-eval."
    return "No durable ppmlx benchmark decision from this batch."


def _extract_session_context(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "system" and "Compacted local session context" in str(message.get("content", "")):
            return str(message.get("content", ""))
    return ""


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _normalize(text: str) -> str:
    return " ".join(str(text).lower().split())

"""Checkpointed quality benchmark for rolling context + temporal graph.

The benchmark turns a real long transcript into prefix/holdout probes:
- split by episodes (default 80% prefix, 20% holdout)
- for each held-out user turn, send prefix + that user turn to a live ppmlx server
- compare the local model answer against the recorded next assistant answer
- score the same five quality layers used by answer-quality eval
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ppmlx.answer_quality import AnswerQualityCase, AnswerQualityEvaluator, AnswerQualityThresholds, match_facts, select_required_facts
from ppmlx.answer_quality_replay import _post_chat, load_session_messages
from ppmlx.context_reducer import ContextBudget, estimate_messages_tokens, group_messages_into_episodes
from ppmlx.trace_replay import TRACE_SCHEMA, compact_replay

Responder = Callable[[list[dict[str, Any]], int, dict[str, Any]], tuple[str, dict[str, Any], float]]


@dataclass
class QualityBenchProbe:
    probe_id: str
    prefix_messages: list[dict[str, Any]]
    user_message: dict[str, Any]
    expected_answer: str
    episode_index: int
    original_prefix_tokens: int
    probe_type: str = "answerable_text"
    classifier_reason: str = ""


@dataclass
class QualityBenchSkippedProbe:
    probe_id: str
    episode_index: int
    probe_type: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "episode_index": self.episode_index,
            "probe_type": self.probe_type,
            "reason": self.reason,
        }


@dataclass
class QualityBenchPreparedProbe:
    probe: QualityBenchProbe
    replay: dict[str, Any]
    source_context: str
    required_facts: list[str]
    raw_required_facts: list[str]
    context_found: list[str]
    context_missing: list[str]
    context_fact_coverage: float
    prefix_found: list[str] = field(default_factory=list)
    prefix_missing: list[str] = field(default_factory=list)
    compaction_lost: list[str] = field(default_factory=list)


@dataclass
class QualityBenchWorkflowProbeResult:
    probe_id: str
    episode_index: int
    probe_type: str
    classifier_reason: str
    passed: bool
    oracle_facts_count: int
    context_found_count: int
    context_missing_count: int
    context_fact_coverage: float
    retrieval_latency_ms: float
    context_items: int
    replay_compression: float
    failure_bucket: str
    expected_answer: str | None = None
    raw_required_facts: list[str] | None = None
    context_found_facts: list[str] | None = None
    context_missing_facts: list[str] | None = None
    source_context: str | None = None
    prefix_found_facts: list[str] | None = None
    prefix_missing_facts: list[str] | None = None
    compaction_lost_facts: list[str] | None = None

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        data = {
            "probe_id": self.probe_id,
            "episode_index": self.episode_index,
            "probe_type": self.probe_type,
            "classifier_reason": self.classifier_reason,
            "passed": self.passed,
            "oracle_facts_count": self.oracle_facts_count,
            "context_found_count": self.context_found_count,
            "context_missing_count": self.context_missing_count,
            "context_fact_coverage": self.context_fact_coverage,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "context_items": self.context_items,
            "replay_compression": self.replay_compression,
            "failure_bucket": self.failure_bucket,
        }
        if include_content:
            data.update({
                "expected_answer": self.expected_answer,
                "raw_required_facts": self.raw_required_facts or [],
                "context_found_facts": self.context_found_facts or [],
                "context_missing_facts": self.context_missing_facts or [],
                "prefix_found_facts": self.prefix_found_facts or [],
                "prefix_missing_facts": self.prefix_missing_facts or [],
                "compaction_lost_facts": self.compaction_lost_facts or [],
                "source_context": self.source_context or "",
            })
        return data


@dataclass
class QualityBenchProbeResult:
    probe_id: str
    passed: bool
    original_prefix_tokens: int
    prompt_tokens: int
    completion_tokens: int
    prompt_compression: float
    replay_compression: float
    context_items: int
    required_facts_count: int
    fact_copy_score: float
    recall: float
    wrong_facts: int
    actionability: float
    grounding: float
    equivalence: float
    latency_sec: float
    missed_count: int
    unsupported_count: int
    context_fact_coverage: float = 0.0
    context_missing_count: int = 0
    failure_bucket: str = "unknown"
    used_extractive_fallback: bool = False
    ablations: dict[str, Any] = field(default_factory=dict)
    probe_type: str = "answerable_text"
    classifier_reason: str = ""
    error: str | None = None
    generated_answer: str | None = None
    expected_answer: str | None = None
    required_facts: list[str] | None = None
    raw_required_facts: list[str] | None = None
    context_found_facts: list[str] | None = None
    context_missing_facts: list[str] | None = None
    source_context: str | None = None
    prefix_found_facts: list[str] | None = None
    prefix_missing_facts: list[str] | None = None
    compaction_lost_facts: list[str] | None = None

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        data = {
            "probe_id": self.probe_id,
            "passed": self.passed,
            "original_prefix_tokens": self.original_prefix_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "prompt_compression": self.prompt_compression,
            "replay_compression": self.replay_compression,
            "context_items": self.context_items,
            "required_facts_count": self.required_facts_count,
            "fact_copy_score": self.fact_copy_score,
            "recall": self.recall,
            "wrong_facts": self.wrong_facts,
            "actionability": self.actionability,
            "grounding": self.grounding,
            "equivalence": self.equivalence,
            "latency_sec": self.latency_sec,
            "missed_count": self.missed_count,
            "unsupported_count": self.unsupported_count,
            "context_fact_coverage": self.context_fact_coverage,
            "context_missing_count": self.context_missing_count,
            "failure_bucket": self.failure_bucket,
            "used_extractive_fallback": self.used_extractive_fallback,
            "ablations": self.ablations,
            "probe_type": self.probe_type,
            "classifier_reason": self.classifier_reason,
            "error": self.error,
        }
        if include_content:
            data.update({
                "generated_answer": self.generated_answer,
                "expected_answer": self.expected_answer,
                "required_facts": self.required_facts or [],
                "raw_required_facts": self.raw_required_facts or [],
                "context_found_facts": self.context_found_facts or [],
                "context_missing_facts": self.context_missing_facts or [],
                "prefix_found_facts": self.prefix_found_facts or [],
                "prefix_missing_facts": self.prefix_missing_facts or [],
                "compaction_lost_facts": self.compaction_lost_facts or [],
                "source_context": self.source_context or "",
            })
        return data


@dataclass
class QualityBenchThresholds:
    min_oracle_recoverable_rate: float = 0.5
    min_context_fact_coverage: float = 0.5
    max_retrieval_p95_ms: float = 100.0

    def to_dict(self) -> dict[str, float]:
        return {
            "min_oracle_recoverable_rate": self.min_oracle_recoverable_rate,
            "min_context_fact_coverage": self.min_context_fact_coverage,
            "max_retrieval_p95_ms": self.max_retrieval_p95_ms,
        }


@dataclass
class QualityBenchReport:
    timestamp: str
    path: str
    source: str
    model: str
    split: float
    total_messages: int
    total_tokens: int
    probes: list[QualityBenchProbeResult] = field(default_factory=list)
    workflow_probes: list[QualityBenchWorkflowProbeResult] = field(default_factory=list)
    skipped_probes: list[QualityBenchSkippedProbe] = field(default_factory=list)
    preflight: dict[str, Any] = field(default_factory=dict)
    thresholds: QualityBenchThresholds = field(default_factory=QualityBenchThresholds)
    preflight_only: bool = False

    @property
    def context_passed(self) -> bool:
        return bool(self.preflight.get("context_passed"))

    @property
    def answer_passed(self) -> bool:
        ok = [probe for probe in self.probes if not probe.error]
        return bool(ok) and all(probe.passed for probe in ok) and not any(probe.error for probe in self.probes)

    @property
    def passed(self) -> bool:
        return self.context_passed and (self.preflight_only or self.answer_passed)

    def summary(self) -> dict[str, Any]:
        ok = [probe for probe in self.probes if not probe.error]
        answer_score = {
            "evaluated_probes": len(self.probes),
            "ok": len(ok),
            "passed": sum(1 for probe in ok if probe.passed),
            "failed": sum(1 for probe in ok if not probe.passed),
            "errors": sum(1 for probe in self.probes if probe.error),
            "pass_rate": _ratio(sum(1 for probe in ok if probe.passed), len(ok)),
            "avg_fact_copy_score": _avg([probe.fact_copy_score for probe in ok]),
            "avg_recall": _avg([probe.recall for probe in ok]),
            "wrong_facts_total": sum(probe.wrong_facts for probe in ok),
            "avg_actionability": _avg([probe.actionability for probe in ok]),
            "avg_grounding": _avg([probe.grounding for probe in ok]),
            "avg_equivalence": _avg([probe.equivalence for probe in ok]),
            "avg_prompt_compression": _avg([probe.prompt_compression for probe in ok]),
            "avg_replay_compression": _avg([probe.replay_compression for probe in ok]),
            "prompt_tokens_total": sum(probe.prompt_tokens for probe in ok),
            "completion_tokens_total": sum(probe.completion_tokens for probe in ok),
        }
        context_score = {
            "candidate_probes": self.preflight.get("candidate_probes", 0),
            "oracle_recoverable_probes": self.preflight.get("oracle_recoverable_probes", 0),
            "oracle_unavailable_probes": self.preflight.get("oracle_unavailable_probes", 0),
            "oracle_recoverable_rate": self.preflight.get("oracle_recoverable_rate", 0.0),
            "avg_context_fact_coverage": self.preflight.get("avg_context_fact_coverage", 0.0),
            "min_context_fact_coverage": self.preflight.get("min_context_fact_coverage", 0.0),
            "retrieval_latency_ms_p95": self.preflight.get("retrieval_latency_ms_p95", 0.0),
            "status": self.preflight.get("context_status", "pass" if self.context_passed else "fail"),
            "not_applicable": self.preflight.get("not_applicable", False),
            "failure_buckets": self.preflight.get("failure_buckets", {}),
            "passed": self.context_passed,
        }
        workflow_score = _workflow_summary(self.workflow_probes, thresholds=self.thresholds)
        return {
            "probes": len(self.probes),
            "ok": len(ok),
            "passed": answer_score["passed"],
            "failed": answer_score["failed"],
            "errors": answer_score["errors"],
            "skipped": len(self.skipped_probes),
            "skipped_by_type": _count_by_type([probe.probe_type for probe in self.skipped_probes]),
            "failure_buckets": _count_by_type([probe.failure_bucket for probe in ok]),
            "avg_fact_copy_score": answer_score["avg_fact_copy_score"],
            "avg_recall": answer_score["avg_recall"],
            "wrong_facts_total": answer_score["wrong_facts_total"],
            "avg_actionability": answer_score["avg_actionability"],
            "avg_grounding": answer_score["avg_grounding"],
            "avg_equivalence": answer_score["avg_equivalence"],
            "avg_context_fact_coverage": context_score["avg_context_fact_coverage"],
            "avg_prompt_compression": answer_score["avg_prompt_compression"],
            "avg_replay_compression": answer_score["avg_replay_compression"],
            "prompt_tokens_total": answer_score["prompt_tokens_total"],
            "completion_tokens_total": answer_score["completion_tokens_total"],
            "preflight": self.preflight,
            "context_score": context_score,
            "answer_score": answer_score,
            "workflow_score": workflow_score,
            "context_passed": self.context_passed,
            "answer_passed": self.answer_passed,
            "preflight_only": self.preflight_only,
        }

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "path": self.path,
            "source": self.source,
            "model": self.model,
            "split": self.split,
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "thresholds": self.thresholds.to_dict(),
            "preflight_only": self.preflight_only,
            "context_passed": self.context_passed,
            "answer_passed": self.answer_passed,
            "preflight": self.preflight,
            "summary": self.summary(),
            "probes": [probe.to_dict(include_content=include_content) for probe in self.probes],
            "workflow_probes": [probe.to_dict(include_content=include_content) for probe in self.workflow_probes],
            "skipped_probes": [probe.to_dict() for probe in self.skipped_probes],
        }


def build_quality_probes(
    messages: list[dict[str, Any]],
    *,
    split: float = 0.8,
    max_probes: int = 5,
    min_expected_chars: int = 40,
    include_probe_types: tuple[str, ...] = ("answerable_text",),
) -> tuple[list[QualityBenchProbe], list[QualityBenchSkippedProbe]]:
    """Build holdout probes from a transcript without leaking expected answers."""
    if not 0.1 <= split <= 0.95:
        raise ValueError("split must be between 0.1 and 0.95")
    system_messages = [message for message in messages if message.get("role") == "system"]
    non_system = [message for message in messages if message.get("role") != "system"]
    episodes = group_messages_into_episodes(non_system)
    if len(episodes) < 2:
        return [], []
    total_tokens = estimate_messages_tokens(non_system)
    split_tokens = max(1, int(total_tokens * split))
    prefix_episode_count = 0
    running = 0
    for episode in episodes:
        if prefix_episode_count < len(episodes) - 1 and running + episode.tokens <= split_tokens:
            running += episode.tokens
            prefix_episode_count += 1
        else:
            break
    if prefix_episode_count >= len(episodes):
        prefix_episode_count = len(episodes) - 1

    probes: list[QualityBenchProbe] = []
    skipped: list[QualityBenchSkippedProbe] = []
    for episode_index in range(prefix_episode_count, len(episodes)):
        episode = episodes[episode_index]
        user_message = _first_user_message(episode.messages)
        expected = _first_assistant_after_user(episode.messages)
        probe_id = f"probe-{episode_index}"
        if not user_message or not expected or len(expected.strip()) < min_expected_chars:
            skipped.append(QualityBenchSkippedProbe(
                probe_id=probe_id,
                episode_index=episode_index,
                probe_type="ambiguous_skip",
                reason="missing user/assistant pair or expected answer too short",
            ))
            continue
        probe_type, reason = classify_probe(_message_text(user_message), expected)
        if probe_type not in include_probe_types:
            skipped.append(QualityBenchSkippedProbe(
                probe_id=probe_id,
                episode_index=episode_index,
                probe_type=probe_type,
                reason=reason,
            ))
            continue
        prior_messages = [*system_messages]
        for prior in episodes[:episode_index]:
            prior_messages.extend(prior.messages)
        original_prefix_tokens = estimate_messages_tokens([*prior_messages, user_message])
        probes.append(QualityBenchProbe(
            probe_id=probe_id,
            prefix_messages=prior_messages,
            user_message=user_message,
            expected_answer=expected,
            episode_index=episode_index,
            original_prefix_tokens=original_prefix_tokens,
            probe_type=probe_type,
            classifier_reason=reason,
        ))
        if len(probes) >= max_probes:
            break
    return probes, skipped


def run_quality_bench(
    *,
    path: Path | str,
    source: str = "auto",
    base_url: str = "http://127.0.0.1:6767/v1",
    model: str,
    split: float = 0.8,
    max_probes: int = 5,
    max_tokens: int = 260,
    include_probe_types: tuple[str, ...] = ("answerable_text",),
    timeout_sec: float = 600,
    include_content: bool = False,
    preflight_only: bool = False,
    thresholds: QualityBenchThresholds | None = None,
    compact_threshold_tokens: int = 1_500,
    hot_tail_tokens: int = 1_200,
    session_context_tokens: int = 4_000,
    max_context_items: int = 80,
    extraction_model: str = "gemma-4-e2b",
    hybrid_extraction: bool = True,
    include_workflow_score: bool = True,
    workflow_probe_types: tuple[str, ...] = ("tool_action_required", "code_edit_required"),
    extractive_fallback: bool = False,
    responder: Responder | None = None,
) -> QualityBenchReport:
    resolved_source, messages = load_session_messages(path, source=source)
    probes, skipped = build_quality_probes(
        messages,
        split=split,
        max_probes=max_probes,
        include_probe_types=include_probe_types,
    )
    workflow_candidates: list[QualityBenchProbe] = []
    if include_workflow_score and workflow_probe_types:
        workflow_candidates, _ = build_quality_probes(
            messages,
            split=split,
            max_probes=max_probes,
            include_probe_types=workflow_probe_types,
        )
    results: list[QualityBenchProbeResult] = []
    workflow_results: list[QualityBenchWorkflowProbeResult] = []
    all_skipped = list(skipped)
    prepared_probes: list[QualityBenchPreparedProbe] = []
    all_prepared_diagnostics: list[QualityBenchPreparedProbe] = []
    oracle_unavailable = 0
    for probe in probes:
        prepared = prepare_quality_probe(
            probe,
            model=model,
            compact_threshold_tokens=compact_threshold_tokens,
            hot_tail_tokens=hot_tail_tokens,
            session_context_tokens=session_context_tokens,
            max_context_items=max_context_items,
            extraction_model=extraction_model,
            hybrid_extraction=hybrid_extraction,
        )
        all_prepared_diagnostics.append(prepared)
        if not prepared.required_facts:
            oracle_unavailable += 1
            lost_in_compaction = bool(prepared.compaction_lost)
            all_skipped.append(QualityBenchSkippedProbe(
                probe_id=probe.probe_id,
                episode_index=probe.episode_index,
                probe_type="oracle_lost_in_compaction" if lost_in_compaction else "oracle_unavailable_in_context",
                reason=(
                    "expected-answer oracle facts were present in the full prefix but missing from compact/replay context"
                    if lost_in_compaction
                    else "no expected-answer oracle facts are recoverable from compact/replay context"
                ),
            ))
            continue
        prepared_probes.append(prepared)
    thresholds = thresholds or QualityBenchThresholds()
    oracle_recoverable_rate = _ratio(len(prepared_probes), len(probes))
    avg_context_fact_coverage = _avg([probe.context_fact_coverage for probe in prepared_probes])
    min_context_fact_coverage = min((probe.context_fact_coverage for probe in prepared_probes), default=0.0)
    retrieval_p95 = _percentile([float(probe.replay.get("retrieval_latency_ms") or 0.0) for probe in prepared_probes], 95)
    context_status = _context_gate_status(
        candidate_probes=len(probes),
        oracle_recoverable_rate=oracle_recoverable_rate,
        avg_context_fact_coverage=avg_context_fact_coverage,
        retrieval_p95=retrieval_p95,
        thresholds=thresholds,
    )
    context_passed = context_status == "pass"
    for workflow_probe in workflow_candidates:
        workflow_prepared = prepare_workflow_probe(
            workflow_probe,
            model=model,
            compact_threshold_tokens=compact_threshold_tokens,
            hot_tail_tokens=hot_tail_tokens,
            session_context_tokens=session_context_tokens,
            max_context_items=max_context_items,
            extraction_model=extraction_model,
            hybrid_extraction=hybrid_extraction,
        )
        workflow_results.append(_workflow_probe_result(workflow_prepared, thresholds=thresholds, include_content=include_content))

    retrieval_latencies = [float(probe.replay.get("retrieval_latency_ms") or 0.0) for probe in prepared_probes]
    preflight_failure_buckets = _context_gate_failure_buckets(
        candidate_probes=len(probes),
        oracle_recoverable_rate=oracle_recoverable_rate,
        avg_context_fact_coverage=avg_context_fact_coverage,
        retrieval_p95=retrieval_p95,
        thresholds=thresholds,
    )
    preflight = {
        "candidate_probes": len(probes),
        "oracle_recoverable_probes": len(prepared_probes),
        "oracle_unavailable_probes": oracle_unavailable,
        "oracle_recoverable_rate": round(oracle_recoverable_rate, 4),
        "classifier_skipped_probes": len(skipped),
        "skipped_by_type": _count_by_type([probe.probe_type for probe in all_skipped]),
        "context_status": context_status,
        "not_applicable": context_status == "not_applicable",
        "failure_buckets": preflight_failure_buckets,
        "oracle_prefix_found_facts": sum(len(probe.prefix_found) for probe in all_prepared_diagnostics),
        "oracle_prefix_missing_facts": sum(len(probe.prefix_missing) for probe in all_prepared_diagnostics),
        "oracle_compaction_lost_facts": sum(len(probe.compaction_lost) for probe in all_prepared_diagnostics),
        "avg_context_fact_coverage": avg_context_fact_coverage,
        "min_context_fact_coverage": min_context_fact_coverage,
        "retrieval_latency_ms_avg": _avg(retrieval_latencies),
        "retrieval_latency_ms_p95": round(retrieval_p95, 4),
        "retrieval_latency_ms_max": round(max(retrieval_latencies, default=0.0), 4),
        "context_passed": context_passed,
    }
    if not preflight_only:
        for prepared in prepared_probes:
            results.append(run_quality_probe(
                prepared.probe,
                base_url=base_url,
                model=model,
                source=resolved_source,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                include_content=include_content,
                responder=responder,
                prepared=prepared,
                extractive_fallback=extractive_fallback,
            ))
    return QualityBenchReport(
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        path=str(path),
        source=resolved_source,
        model=model,
        split=split,
        total_messages=len(messages),
        total_tokens=estimate_messages_tokens(messages),
        probes=results,
        workflow_probes=workflow_results,
        skipped_probes=all_skipped,
        preflight=preflight,
        thresholds=thresholds,
        preflight_only=preflight_only,
    )


def _workflow_probe_result(
    prepared: QualityBenchPreparedProbe,
    *,
    thresholds: QualityBenchThresholds,
    include_content: bool,
) -> QualityBenchWorkflowProbeResult:
    probe = prepared.probe
    oracle_count = len(prepared.raw_required_facts)
    found_count = len(prepared.context_found)
    coverage = round(prepared.context_fact_coverage, 4)
    passed = oracle_count > 0 and found_count > 0 and coverage >= thresholds.min_context_fact_coverage
    if passed:
        bucket = "passed"
    elif oracle_count <= 0:
        bucket = "workflow_no_oracle_facts"
    elif found_count <= 0:
        bucket = "workflow_context_missing"
    else:
        bucket = "workflow_low_context_coverage"
    return QualityBenchWorkflowProbeResult(
        probe_id=probe.probe_id,
        episode_index=probe.episode_index,
        probe_type=probe.probe_type,
        classifier_reason=probe.classifier_reason,
        passed=passed,
        oracle_facts_count=oracle_count,
        context_found_count=found_count,
        context_missing_count=len(prepared.context_missing),
        context_fact_coverage=coverage,
        retrieval_latency_ms=float(prepared.replay.get("retrieval_latency_ms") or 0.0),
        context_items=int(prepared.replay.get("context_items") or 0),
        replay_compression=float(prepared.replay.get("compression_ratio") or 0.0),
        failure_bucket=bucket,
        expected_answer=probe.expected_answer if include_content else None,
        raw_required_facts=prepared.raw_required_facts if include_content else None,
        context_found_facts=prepared.context_found if include_content else None,
        context_missing_facts=prepared.context_missing if include_content else None,
        source_context=prepared.source_context if include_content else None,
        prefix_found_facts=prepared.prefix_found if include_content else None,
        prefix_missing_facts=prepared.prefix_missing if include_content else None,
        compaction_lost_facts=prepared.compaction_lost if include_content else None,
    )


def _context_gate_status(
    *,
    candidate_probes: int,
    oracle_recoverable_rate: float,
    avg_context_fact_coverage: float,
    retrieval_p95: float,
    thresholds: QualityBenchThresholds,
) -> str:
    if candidate_probes <= 0:
        return "not_applicable"
    if _context_gate_failure_buckets(
        candidate_probes=candidate_probes,
        oracle_recoverable_rate=oracle_recoverable_rate,
        avg_context_fact_coverage=avg_context_fact_coverage,
        retrieval_p95=retrieval_p95,
        thresholds=thresholds,
    ):
        return "fail"
    return "pass"


def _context_gate_failure_buckets(
    *,
    candidate_probes: int,
    oracle_recoverable_rate: float,
    avg_context_fact_coverage: float,
    retrieval_p95: float,
    thresholds: QualityBenchThresholds,
) -> dict[str, int]:
    buckets: list[str] = []
    if candidate_probes <= 0:
        buckets.append("not_applicable_no_candidate_probes")
    if candidate_probes > 0 and oracle_recoverable_rate < thresholds.min_oracle_recoverable_rate:
        buckets.append("low_oracle_recoverable")
    if candidate_probes > 0 and avg_context_fact_coverage < thresholds.min_context_fact_coverage:
        buckets.append("low_context_fact_coverage")
    if retrieval_p95 > thresholds.max_retrieval_p95_ms:
        buckets.append("slow_retrieval")
    return _count_by_type(buckets)


def _workflow_summary(probes: list[QualityBenchWorkflowProbeResult], *, thresholds: QualityBenchThresholds) -> dict[str, Any]:
    total = len(probes)
    passed = sum(1 for probe in probes if probe.passed)
    retrieval_latencies = [probe.retrieval_latency_ms for probe in probes]
    return {
        "probes": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": _ratio(passed, total),
        "avg_context_fact_coverage": _avg([probe.context_fact_coverage for probe in probes]),
        "min_context_fact_coverage": min((probe.context_fact_coverage for probe in probes), default=0.0),
        "oracle_facts_total": sum(probe.oracle_facts_count for probe in probes),
        "context_found_total": sum(probe.context_found_count for probe in probes),
        "context_missing_total": sum(probe.context_missing_count for probe in probes),
        "retrieval_latency_ms_avg": _avg(retrieval_latencies),
        "retrieval_latency_ms_p95": round(_percentile(retrieval_latencies, 95), 4),
        "retrieval_latency_ms_max": round(max(retrieval_latencies, default=0.0), 4),
        "failure_buckets": _count_by_type([probe.failure_bucket for probe in probes]),
        "passed_threshold": thresholds.min_context_fact_coverage,
    }


def prepare_quality_probe(
    probe: QualityBenchProbe,
    *,
    model: str,
    compact_threshold_tokens: int = 1_500,
    hot_tail_tokens: int = 1_200,
    session_context_tokens: int = 4_000,
    max_context_items: int = 80,
    extraction_model: str = "gemma-4-e2b",
    hybrid_extraction: bool = True,
) -> QualityBenchPreparedProbe:
    """Prepare replay/oracle facts, keeping only facts visible in compact context.

    The recorded next assistant answer often contains new synthesis or tool-loop
    actions that are not present in the prefix. Those are useful diagnostics but
    not fair recall requirements for a text-only compact-context probe.
    """
    messages = [*probe.prefix_messages, probe.user_message]
    replay = _probe_replay(
        messages,
        model=model,
        probe_id=probe.probe_id,
        compact_threshold_tokens=compact_threshold_tokens,
        hot_tail_tokens=hot_tail_tokens,
        session_context_tokens=session_context_tokens,
        max_context_items=max_context_items,
        extraction_model=extraction_model,
        hybrid_extraction=hybrid_extraction,
    )
    source_context = replay.get("reduced_context") or replay.get("session_context") or ""
    raw_required = select_required_facts(
        source_context=probe.expected_answer,
        question=_message_text(probe.user_message),
        reference_answer=probe.expected_answer,
        max_facts=8,
    )
    if not raw_required:
        raw_required = select_required_facts(
            source_context=source_context,
            question=_message_text(probe.user_message),
            reference_answer=probe.expected_answer,
            max_facts=8,
        )
    raw_required = [fact for fact in raw_required if _is_informative_oracle_fact(fact)]
    prefix_context = _messages_context_text(messages)
    return _prepare_probe_from_raw_required(
        probe=probe,
        replay=replay,
        source_context=source_context,
        prefix_context=prefix_context,
        raw_required=raw_required,
    )


def prepare_workflow_probe(
    probe: QualityBenchProbe,
    *,
    model: str,
    compact_threshold_tokens: int = 1_500,
    hot_tail_tokens: int = 1_200,
    session_context_tokens: int = 4_000,
    max_context_items: int = 80,
    extraction_model: str = "gemma-4-e2b",
    hybrid_extraction: bool = True,
) -> QualityBenchPreparedProbe:
    """Prepare workflow/action continuity probes from prior actionable state.

    For generic action turns ("działaj", "continue", "do it"), the fair oracle is
    the previous plan/todo/validation/blocker state, not the hidden next assistant
    response text.
    """
    messages = [*probe.prefix_messages, probe.user_message]
    replay = _probe_replay(
        messages,
        model=model,
        probe_id=probe.probe_id,
        compact_threshold_tokens=compact_threshold_tokens,
        hot_tail_tokens=hot_tail_tokens,
        session_context_tokens=session_context_tokens,
        max_context_items=max_context_items,
        extraction_model=extraction_model,
        hybrid_extraction=hybrid_extraction,
    )
    source_context = replay.get("reduced_context") or replay.get("session_context") or ""
    prefix_context = _messages_context_text(probe.prefix_messages)
    raw_required = _select_workflow_required_facts(probe.prefix_messages, max_facts=8)
    if not raw_required:
        raw_required = _select_workflow_required_facts_from_text(source_context, max_facts=8)
    return _prepare_probe_from_raw_required(
        probe=probe,
        replay=replay,
        source_context=source_context,
        prefix_context=prefix_context,
        raw_required=raw_required,
    )


def _prepare_probe_from_raw_required(
    *,
    probe: QualityBenchProbe,
    replay: dict[str, Any],
    source_context: str,
    prefix_context: str,
    raw_required: list[str],
) -> QualityBenchPreparedProbe:
    raw_required = [fact for fact in raw_required if _is_informative_oracle_fact(fact)]
    context_found, context_missing = _map_required_facts_to_source(source_context, raw_required)
    prefix_found, prefix_missing = _raw_facts_found_in_source(prefix_context, raw_required)
    compaction_lost = [fact for fact in prefix_found if _norm(fact) in {_norm(item) for item in context_missing}]
    if not context_found and raw_required:
        required_facts: list[str] = []
    else:
        required_facts = context_found or raw_required
    return QualityBenchPreparedProbe(
        probe=probe,
        replay=replay,
        source_context=source_context,
        required_facts=required_facts,
        raw_required_facts=raw_required,
        context_found=context_found,
        context_missing=context_missing,
        context_fact_coverage=_ratio(len(context_found), len(raw_required)),
        prefix_found=prefix_found,
        prefix_missing=prefix_missing,
        compaction_lost=compaction_lost,
    )


def _is_informative_oracle_fact(fact: str) -> bool:
    cleaned = str(fact or "").strip().strip("`*_ ")
    if len(cleaned) < 8:
        return False
    if re.fullmatch(r"[─━\-=|\s]+", cleaned):
        return False
    norm = _norm(cleaned)
    if norm in {"kontynuuj", "continue", "działaj", "dzialaj", "źródła seedów", "zrodla seedow"}:
        return False
    if any(marker in norm for marker in ("pnpm", "npm", "yarn", "pytest", "ruff", "eslint", "build", "origin/")):
        return True
    tokens = re.findall(r"[a-ząćęłńóśźż0-9_./:-]+", norm)
    if len(tokens) <= 2 and not any(char.isdigit() for char in cleaned) and not any(char in cleaned for char in ("/", ".", "_", "-")):
        return False
    return True


def _messages_context_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for message in messages:
        role = str(message.get("role") or "message")
        text = _message_text(message)
        if text.strip():
            parts.append(f"[{role}] {text}")
    return "\n".join(parts)


def _raw_facts_found_in_source(source_context: str, raw_required: list[str]) -> tuple[list[str], list[str]]:
    if not raw_required:
        return [], []
    _, missing = _map_required_facts_to_source(source_context, raw_required)
    missing_norm = {_norm(item) for item in missing}
    found = [fact for fact in raw_required if _norm(fact) not in missing_norm]
    return found, missing


def _select_workflow_required_facts(messages: list[dict[str, Any]], *, max_facts: int) -> list[str]:
    selected: list[str] = []
    for message in reversed(messages):
        role = str(message.get("role") or "").lower()
        if role not in {"assistant", "user", "developer"}:
            continue
        for fact in _select_workflow_required_facts_from_text(_message_text(message), max_facts=max_facts):
            if _norm(fact) not in {_norm(item) for item in selected}:
                selected.append(fact)
            if len(selected) >= max_facts:
                return list(reversed(selected))
    return list(reversed(selected))


def _select_workflow_required_facts_from_text(text: str, *, max_facts: int) -> list[str]:
    candidates: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip().lstrip("-•* ").strip()
        if not line:
            continue
        for part in _workflow_fact_parts(line):
            cleaned = _clean_workflow_fact(part)
            if cleaned and _looks_like_workflow_fact(cleaned) and _norm(cleaned) not in {_norm(item) for item in candidates}:
                candidates.append(cleaned)
                if len(candidates) >= max_facts:
                    return candidates
    return candidates


def _workflow_fact_parts(line: str) -> list[str]:
    if len(line) <= 260:
        return [line]
    parts = re.split(r"(?<=[.!?])\s+", line)
    return [part for part in parts if part.strip()]


def _clean_workflow_fact(text: str) -> str:
    cleaned = re.sub(r"^\s*(assistant|user|developer)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*(todo|task|next action|next step|next|current task|current work|validation|walidacja|blocker|commit)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" `*_-.;:")


def _looks_like_workflow_fact(text: str) -> bool:
    lowered = _norm(text)
    if len(lowered) < 8:
        return False
    markers = (
        "todo", "next", "current task", "current work", "validation", "walidacja", "commit", "origin/",
        "changed", "updated", "modified", "patched", "created", "file", "blocker", "blocked", "failing",
        "failed", "error", "fix", "deploy", "apply", "run", "rerun", "test", "pytest", "ruff", "eslint",
        "pnpm", "npm", "yarn", "uv run", "build", "✅",
    )
    return any(marker in lowered for marker in markers) or bool(re.search(r"\b[0-9a-f]{7,40}\b", text))


def _map_required_facts_to_source(source_context: str, raw_required: list[str]) -> tuple[list[str], list[str]]:
    """Map oracle facts to the concrete fact strings available in retrieved context.

    The oracle/reference answer often paraphrases the same state differently from
    the compacted graph bullets.  Answer scoring should require the model to use
    the retrieved fact text, not to reproduce hidden reference-answer wording.
    """
    if not raw_required:
        return [], []
    candidates = _source_fact_candidates(source_context)
    found: list[str] = []
    missing: list[str] = []
    used: set[str] = set()
    for raw_fact in raw_required:
        raw_tokens = _fact_tokens_for_mapping(raw_fact)
        if len(raw_tokens) <= 4 and match_facts(source_context, [raw_fact])[0]:
            found.append(raw_fact)
            continue
        scored: list[tuple[float, int, str]] = []
        for candidate in candidates:
            if candidate in used:
                continue
            if _candidate_matches_raw_fact(candidate, raw_fact):
                scored.append((_fact_overlap(raw_fact, candidate), -len(candidate), candidate))
        if scored:
            scored.sort(reverse=True)
            chosen = scored[0][2]
            found.append(chosen)
            used.add(chosen)
        elif match_facts(source_context, [raw_fact])[0]:
            found.append(raw_fact)
        else:
            missing.append(raw_fact)
    return found, missing


def _source_fact_candidates(source_context: str) -> list[str]:
    candidates: list[str] = []
    for raw_line in source_context.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("["):
            continue
        if line.lower().startswith("question ") or line.endswith("?"):
            continue
        if line.startswith("- "):
            line = line[2:]
        line = line.split(" [source:", 1)[0].split(" (scope=", 1)[0].strip().rstrip(".")
        line = re.sub(r"^[A-Za-z0-9_.:-]+\s+(constraint|decision|todo):\s*", "", line, flags=re.IGNORECASE)
        for fragment in _source_fact_line_fragments(line):
            candidates.append(fragment)
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _source_fact_line_fragments(line: str) -> list[str]:
    if 8 <= len(line) <= 260:
        return [line]
    if len(line) < 8:
        return []
    fragments: list[str] = []
    for fragment in re.split(r"(?<=[.!?])\s+|\s+[•·]\s+|\s+-\s+", line):
        cleaned = fragment.strip().strip("-•· ").rstrip(".")
        if 8 <= len(cleaned) <= 260:
            fragments.append(cleaned)
    return fragments


def _candidate_matches_raw_fact(candidate: str, raw_fact: str) -> bool:
    if match_facts(candidate, [raw_fact])[0] or match_facts(raw_fact, [candidate])[0]:
        return True
    candidate_norm = _norm(candidate)
    if any(token not in candidate_norm for token in _important_fact_tokens(raw_fact)):
        return False
    left_tokens = set(_fact_tokens_for_mapping(raw_fact))
    right_tokens = set(_fact_tokens_for_mapping(candidate))
    common = len(left_tokens & right_tokens)
    return common >= 3 and _fact_overlap(raw_fact, candidate) >= 0.18


def _important_fact_tokens(text: str) -> list[str]:
    important: list[str] = []
    for token in re.findall(r"[a-ząćęłńóśźż0-9_./:-]+", _norm(text)):
        if any(char.isdigit() for char in token) or any(char in token for char in ("_", "/", ".")):
            important.append(token.strip(".,;:()[]{}"))
    return [token for token in important if token]


def _fact_overlap(left: str, right: str) -> float:
    left_tokens = set(_fact_tokens_for_mapping(left))
    right_tokens = set(_fact_tokens_for_mapping(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _fact_tokens_for_mapping(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-ząćęłńóśźż0-9_./:-]+", _norm(text)) if len(token) > 1]


def run_quality_probe(
    probe: QualityBenchProbe,
    *,
    base_url: str,
    model: str,
    source: str,
    max_tokens: int,
    timeout_sec: float,
    include_content: bool = False,
    responder: Responder | None = None,
    prepared: QualityBenchPreparedProbe | None = None,
    extractive_fallback: bool = False,
) -> QualityBenchProbeResult:
    metadata = {"app_id": "quality-bench", "project_id": f"quality-bench-{source}", "session_id": probe.probe_id}
    started = time.time()
    try:
        prepared = prepared or prepare_quality_probe(probe, model=model)
        replay_messages = prepared.replay.get("reduced_messages") if isinstance(prepared.replay, dict) else None
        if isinstance(replay_messages, list) and replay_messages:
            messages = [_quality_probe_system_message(), *replay_messages]
        else:
            messages = [_quality_probe_system_message(), *probe.prefix_messages, probe.user_message]
        if responder:
            generated, usage, latency = responder(messages, max_tokens, metadata)
        else:
            data, latency = _post_chat(
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                metadata=metadata,
                timeout_sec=timeout_sec,
            )
            generated = _assistant_content(data)
            usage = data.get("usage") or {}
        replay = prepared.replay
        source_context = prepared.source_context
        required_facts = prepared.required_facts
        context_found = prepared.context_found
        context_missing = prepared.context_missing
        context_fact_coverage = prepared.context_fact_coverage
        question = _message_text(probe.user_message)
        expected_actions = _expected_actions_for_probe(question=question, expected_answer=probe.expected_answer)
        evaluator = AnswerQualityEvaluator(AnswerQualityThresholds(min_actionability=_min_actionability(question, probe.expected_answer)))
        quality, fact_copy_score = _evaluate_generated_answer(
            generated=generated,
            probe=probe,
            question=question,
            source_context=source_context,
            required_facts=required_facts,
            expected_actions=expected_actions,
            evaluator=evaluator,
        )
        used_extractive_fallback = False
        if extractive_fallback and required_facts and not quality.passed:
            fallback_answer = _extractive_answer_from_context(question=question, required_facts=required_facts)
            fallback_quality, fallback_copy_score = _evaluate_generated_answer(
                generated=fallback_answer,
                probe=probe,
                question=question,
                source_context=source_context,
                required_facts=required_facts,
                expected_actions=expected_actions,
                evaluator=evaluator,
            )
            if _fallback_is_better(quality, fallback_quality):
                generated = fallback_answer
                quality = fallback_quality
                fact_copy_score = fallback_copy_score
                used_extractive_fallback = True
        failure_bucket = _failure_bucket(
            passed=quality.passed,
            required_facts=required_facts,
            quality=quality,
            thresholds=evaluator.thresholds,
        )
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_compression = round(probe.original_prefix_tokens / max(prompt_tokens, 1), 4)
        return QualityBenchProbeResult(
            probe_id=probe.probe_id,
            passed=quality.passed,
            original_prefix_tokens=probe.original_prefix_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_compression=prompt_compression,
            replay_compression=float(replay.get("compression_ratio") or 0.0),
            context_items=int(replay.get("context_items") or 0),
            required_facts_count=len(required_facts),
            fact_copy_score=fact_copy_score,
            recall=quality.recall,
            wrong_facts=quality.wrong_fact_count,
            actionability=float(quality.actionability),
            grounding=quality.grounding,
            equivalence=quality.equivalence_to_full,
            latency_sec=float(latency),
            missed_count=len(quality.required_missed),
            unsupported_count=len(quality.unsupported_terms),
            context_fact_coverage=round(context_fact_coverage, 4),
            context_missing_count=len(context_missing),
            failure_bucket=failure_bucket,
            used_extractive_fallback=used_extractive_fallback,
            ablations=_ablation_diagnostics(replay),
            probe_type=probe.probe_type,
            classifier_reason=probe.classifier_reason,
            generated_answer=generated if include_content else None,
            expected_answer=probe.expected_answer if include_content else None,
            required_facts=required_facts if include_content else None,
            raw_required_facts=prepared.raw_required_facts if include_content else None,
            context_found_facts=context_found if include_content else None,
            context_missing_facts=context_missing if include_content else None,
            prefix_found_facts=prepared.prefix_found if include_content else None,
            prefix_missing_facts=prepared.prefix_missing if include_content else None,
            compaction_lost_facts=prepared.compaction_lost if include_content else None,
            source_context=source_context if include_content else None,
        )
    except Exception as exc:
        return QualityBenchProbeResult(
            probe_id=probe.probe_id,
            passed=False,
            original_prefix_tokens=probe.original_prefix_tokens,
            prompt_tokens=0,
            completion_tokens=0,
            prompt_compression=0.0,
            replay_compression=0.0,
            context_items=0,
            required_facts_count=0,
            fact_copy_score=0.0,
            recall=0.0,
            wrong_facts=0,
            actionability=0.0,
            grounding=0.0,
            equivalence=0.0,
            latency_sec=round(time.time() - started, 3),
            missed_count=0,
            unsupported_count=0,
            failure_bucket="error",
            probe_type=probe.probe_type,
            classifier_reason=probe.classifier_reason,
            error=repr(exc)[:1000],
        )


def _evaluate_generated_answer(
    *,
    generated: str,
    probe: QualityBenchProbe,
    question: str,
    source_context: str,
    required_facts: list[str],
    expected_actions: list[str],
    evaluator: AnswerQualityEvaluator,
) -> tuple[Any, float]:
    case = AnswerQualityCase(
        case_id=probe.probe_id,
        question=question,
        source_context=source_context,
        compact_answer=generated,
        full_context_answer=probe.expected_answer,
        required_facts=required_facts,
        forbidden_facts=[
            "cannot access earlier context",
            "not enough context",
            "secret key leaked",
            "api key leaked",
            "password leaked",
        ],
        expected_actions=expected_actions,
    )
    quality = evaluator.evaluate_case(case)
    return quality, _fact_copy_score(generated, required_facts)


def _fallback_is_better(current: Any, fallback: Any) -> bool:
    return (
        fallback.passed
        or fallback.recall > current.recall
        or (fallback.recall == current.recall and fallback.grounding > current.grounding)
        or (fallback.recall == current.recall and fallback.grounding == current.grounding and fallback.equivalence_to_full > current.equivalence_to_full)
    )


def _extractive_answer_from_context(*, question: str, required_facts: list[str]) -> str:
    if not required_facts:
        return "The required facts are not available in the retrieved context."
    if len(required_facts) == 1:
        return required_facts[0]
    return "\n".join(f"- {fact}" for fact in required_facts)


def _expected_actions_for_probe(*, question: str, expected_answer: str) -> list[str]:
    text = _norm(f"{question} {expected_answer}")
    actions: list[str] = []
    if any(marker in text for marker in ("next", "todo", "follow up", "follow-up", "następ", "nastep")):
        actions.append("next")
    return actions


def _min_actionability(question: str, expected_answer: str) -> int:
    """Status/recall probes should not fail only because no next-step was needed."""
    return 3 if _expected_actions_for_probe(question=question, expected_answer=expected_answer) else 2


def _quality_probe_system_message() -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are answering a memory quality benchmark probe. Use ONLY the visible conversation and the "
            "Compacted local session context / Retrieved facts. Do not use outside knowledge. "
            "Answer directly and concisely. Copy exact identifiers, filenames, model names, numbers, decisions, "
            "and todos from the retrieved context when they are relevant. If a fact is not in the retrieved "
            "context, say it is not available instead of guessing. Do not add generic caveats or claim you "
            "cannot access earlier context."
        ),
    }


def _fact_copy_score(answer: str, required_facts: list[str]) -> float:
    if not required_facts:
        return 1.0
    found, _ = match_facts(answer, required_facts)
    return round(len(found) / max(len(required_facts), 1), 4)


def _failure_bucket(
    *,
    passed: bool,
    required_facts: list[str],
    quality: Any,
    thresholds: Any,
) -> str:
    if passed:
        return "passed"
    if not required_facts:
        return "no_oracle_facts"
    if quality.recall < thresholds.min_recall:
        return "model_missed_context_facts"
    if quality.grounding < thresholds.min_grounding or quality.unsupported_terms:
        return "ungrounded_generation"
    if quality.actionability < thresholds.min_actionability:
        return "low_actionability"
    if quality.equivalence_to_full < thresholds.min_equivalence_to_full:
        return "low_equivalence"
    return "failed_quality_gate"


def classify_probe(user_text: str, expected_answer: str) -> tuple[str, str]:
    """Classify whether a held-out turn is fair for local text-only answering."""
    user_norm = _norm(user_text)
    expected_norm = _norm(expected_answer)
    if not expected_norm or len(expected_norm) < 40:
        return "ambiguous_skip", "expected answer is too short"
    if expected_norm.startswith("tool call ") or expected_norm.startswith("tool use "):
        return "tool_action_required", "recorded answer is primarily a tool call/action"
    if _looks_like_delegated_action_request(user_norm):
        return "code_edit_required", "user asks assistant to execute prior instructions rather than answer from context"
    if _looks_like_feedback_turn(user_norm):
        return "ambiguous_skip", "user turn is feedback/acknowledgement rather than a standalone answer request"
    if _looks_like_code_or_repo_action(user_norm, expected_norm):
        return "code_edit_required", "user request likely required repository/tool actions"
    if _looks_like_tool_call(expected_answer):
        return "tool_action_required", "recorded answer is primarily a tool call/action"
    if _looks_like_meta_or_ack(expected_norm):
        return "ambiguous_skip", "recorded answer is mostly acknowledgement/meta-process"
    return "answerable_text", "text answer can be compared to recorded answer"


def _ablation_diagnostics(replay: dict[str, Any]) -> dict[str, Any]:
    """Return prompt-shape diagnostics for future ablation runs.

    These are not separate model calls; they expose whether a probe is mostly
    relying on hot tail, graph context, or both so failures can be bucketed.
    """
    original = int(replay.get("original_tokens") or 0)
    reduced = int(replay.get("reduced_tokens") or 0)
    session_context = str(replay.get("session_context") or "")
    reduced_context = str(replay.get("reduced_context") or "")
    session_tokens = int(replay.get("session_context_tokens") or 0)
    approx_hot_tail_tokens = max(0, reduced - session_tokens)
    return {
        "graph_hot": {
            "tokens": reduced,
            "compression": round(original / max(reduced, 1), 4) if original else 0.0,
        },
        "graph_only": {
            "tokens": session_tokens,
            "context_items": int(replay.get("context_items") or 0),
            "has_context": bool(session_context.strip()),
        },
        "hot_tail_only": {
            "tokens": approx_hot_tail_tokens,
            "has_tail": bool(reduced_context.strip()),
        },
    }


def _probe_replay(
    messages: list[dict[str, Any]],
    *,
    model: str,
    probe_id: str,
    compact_threshold_tokens: int = 1_500,
    hot_tail_tokens: int = 1_200,
    session_context_tokens: int = 4_000,
    max_context_items: int = 80,
    extraction_model: str = "gemma-4-e2b",
    hybrid_extraction: bool = True,
) -> dict[str, Any]:
    trace = {
        "schema": TRACE_SCHEMA,
        "events": [
            {
                "event_id": probe_id,
                "endpoint": "/v1/chat/completions",
                "project_id": "quality-bench",
                "session_id": probe_id,
                "model_alias": model,
                "model_repo": model,
                "messages": messages,
                "request": {"messages": messages},
            }
        ],
    }
    budget = ContextBudget(
        mode="compact",
        compact_threshold_tokens=compact_threshold_tokens,
        hot_tail_tokens=hot_tail_tokens,
        session_context_tokens=session_context_tokens,
        max_context_items=max_context_items,
    )
    extractor = _quality_bench_extractor(extraction_model, hybrid_extraction=hybrid_extraction)
    return compact_replay(trace, budget=budget, extractor=extractor).to_dict()


def _quality_bench_extractor(extraction_model: str, *, hybrid_extraction: bool):
    if not hybrid_extraction:
        return None
    from ppmlx.memory_engine import HybridMemoryExtractor, RuleBasedMemoryExtractor
    from ppmlx.memory_extractors import ModelMemoryJsonExtractor

    return HybridMemoryExtractor(
        RuleBasedMemoryExtractor(max_candidates=12),
        ModelMemoryJsonExtractor(model_name=extraction_model, max_candidates=12),
    )


def _first_user_message(messages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for message in messages:
        if message.get("role") in {"user", "developer"} and _message_text(message).strip():
            return message
    return None


def _first_assistant_after_user(messages: list[dict[str, Any]]) -> str | None:
    seen_user = False
    for message in messages:
        if message.get("role") in {"user", "developer"}:
            seen_user = True
            continue
        if seen_user and message.get("role") == "assistant":
            text = _message_text(message)
            if text.strip():
                return text
    return None


def _assistant_content(data: dict[str, Any]) -> str:
    return str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _looks_like_delegated_action_request(user_norm: str) -> bool:
    delegated_markers = (
        "do it for me",
        "do this for me",
        "fix it for me",
        "run it for me",
        "zrob to za mnie",
        "zrób to za mnie",
        "zrob to dla mnie",
        "zrób to dla mnie",
        "zrob to",
        "zrób to",
        "napraw to",
        "odpal to",
    )
    return any(marker in user_norm for marker in delegated_markers)


def _looks_like_feedback_turn(user_norm: str) -> bool:
    if len(user_norm) > 180 or user_norm.endswith("?"):
        return False
    phrase_markers = ("dziala", "działa", "lepiej", "plynniej", "płynniej", "works", "worked")
    if any(re.search(rf"\b{re.escape(marker)}\b", user_norm) for marker in phrase_markers):
        return True
    tokens = set(re.findall(r"[a-ząćęłńóśźż0-9_./:-]+", user_norm))
    return bool(tokens & {"ok", "super", "great", "thanks", "dzieki", "dzięki", "swietnie", "świetnie"})


def _looks_like_tool_call(text: str) -> bool:
    stripped = text.strip()
    lowered = _norm(stripped)
    if lowered.startswith("tool call ") or lowered.startswith("tool use "):
        return True
    if stripped.startswith("{") and any(key in lowered for key in ("tool", "command", "arguments", "tool_name")):
        return True
    tool_markers = (
        "calling tool", "i'll inspect", "i will inspect", "let me inspect", "running ",
        "i'll run", "i will run", "using the", "read the file", "search the repo",
        "i'm checking", "i’m checking", "checking production", "checking prod", "pulling the fresh",
        "pulling fresh", "pulling the current", "error logs now", "i'm applying", "i’m applying",
        "applying the prod",
    )
    return any(marker in lowered for marker in tool_markers) and len(lowered) < 1200


def _looks_like_code_or_repo_action(user_norm: str, expected_norm: str) -> bool:
    action_verbs = (
        "implement", "fix", "edit", "change", "modify", "add", "remove", "refactor",
        "run tests", "test", "commit", "deploy", "build", "inspect", "read", "search",
        "debug", "napraw", "dodaj", "zmień", "zrob", "zrób", "działaj",
    )
    repo_terms = (
        "file", "repo", "test", "pytest", "ruff", "diff", "patch", "commit", "branch",
        "server", "cli", "function", "module", "plik", "testy",
    )
    expected_action = (
        "i'll" in expected_norm or "i will" in expected_norm or "let me" in expected_norm
        or "changed" in expected_norm or "updated" in expected_norm or "edited" in expected_norm
        or "naprawiłem" in expected_norm or "naprawilem" in expected_norm
        or "commit" in expected_norm or "push" in expected_norm or "walidacja" in expected_norm
        or "zaktualizowany" in expected_norm or "pnpm build" in expected_norm
    )
    user_tokens = set(user_norm.replace("-", " ").split())
    action_present = bool(user_tokens & set(action_verbs)) or any(
        token.startswith(("napraw", "zmien", "zmień", "dodaj", "zrob", "zrób"))
        for token in user_tokens
    )
    return action_present and (any(term in user_tokens for term in repo_terms) or expected_action)


def _looks_like_meta_or_ack(text: str) -> bool:
    lowered = _norm(text)
    if len(lowered) < 80:
        tokens = set(re.findall(r"[a-ząćęłńóśźż0-9_./:-]+", lowered))
        if tokens & {"ok", "sure", "done", "jasne"} or "got it" in lowered:
            return True
    meta_markers = (
        "i need to", "the user wants", "i should", "let me", "i'll proceed",
        "i will now", "we need answer", "analysis",
    )
    return any(marker in lowered for marker in meta_markers) and len(lowered) < 500


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def _count_by_type(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _ratio(num: int, den: int) -> float:
    if den <= 0:
        return 1.0
    return num / den


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (percentile / 100)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight

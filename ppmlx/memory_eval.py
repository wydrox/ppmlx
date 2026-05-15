"""Evaluation harness for ppmlx temporal-memory anti-garbage gates.

This module is intentionally independent from a production memory engine.  It
provides a golden corpus, a deterministic reference gate, latency measurement,
and metrics for the failure modes that matter most before memory injection is
allowed into the hot path.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


STATUS_ACTIVE = "active"
STATUS_QUARANTINED = "quarantined"
STATUS_REJECTED = "rejected"
STATUS_DISPUTED = "disputed"

ALLOWED_TYPES = {
    "fact",
    "preference",
    "decision",
    "todo",
    "constraint",
    "entity_note",
    "instruction",
    "relationship",
}

SENSITIVE_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}\b"),
    re.compile(r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b"),
]

SUPERSEDE_SIGNALS = (
    "actually",
    "from now on",
    "instead",
    "no longer",
    "not anymore",
    "prefer concise",
    "supersede",
)

STOPWORDS = {
    "the", "and", "or", "for", "with", "this", "that", "from", "into", "when", "what", "how",
    "should", "would", "could", "about", "only", "user", "users", "keep", "use", "using", "apps",
    "tools", "default", "memory", "local", "project", "session",
}


@dataclass
class MemoryCandidate:
    """A proposed atomic memory before/after validation."""

    id: str
    type: str
    subject: str
    predicate: str
    object: str
    text: str
    scope: str
    confidence: float = 0.0
    source_quote: str | None = None
    salience: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryCandidate":
        return cls(
            id=str(data["id"]),
            type=str(data.get("type", "")),
            subject=str(data.get("subject", "")),
            predicate=str(data.get("predicate", "")),
            object=str(data.get("object", "")),
            text=str(data.get("text", "")),
            scope=str(data.get("scope", "")),
            confidence=float(data.get("confidence", 0.0)),
            source_quote=data.get("source_quote"),
            salience=float(data.get("salience", 1.0)),
            metadata=dict(data.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "text": self.text,
            "scope": self.scope,
            "confidence": self.confidence,
            "source_quote": self.source_quote,
            "salience": self.salience,
            "metadata": self.metadata,
        }

    def memory_key(self) -> tuple[str, str, str, str, str]:
        return (
            _norm(self.type),
            _norm(self.subject),
            _norm(self.predicate),
            _norm(self.object),
            _norm(self.scope),
        )


@dataclass
class ExpectedCandidate:
    id: str
    status: str
    scope: str | None = None
    reasons: list[str] = field(default_factory=list)
    invalidates: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExpectedCandidate":
        return cls(
            id=str(data["id"]),
            status=str(data["status"]),
            scope=data.get("scope"),
            reasons=[str(x) for x in data.get("reasons", [])],
            invalidates=[str(x) for x in data.get("invalidates", [])],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "scope": self.scope,
            "reasons": self.reasons,
            "invalidates": self.invalidates,
        }


@dataclass
class RetrievalExpectation:
    query: str
    expected_ids: list[str] = field(default_factory=list)
    forbidden_ids: list[str] = field(default_factory=list)
    limit: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "RetrievalExpectation | None":
        if not data:
            return None
        return cls(
            query=str(data.get("query", "")),
            expected_ids=[str(x) for x in data.get("expected_ids", [])],
            forbidden_ids=[str(x) for x in data.get("forbidden_ids", [])],
            limit=int(data.get("limit", 8)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "expected_ids": self.expected_ids,
            "forbidden_ids": self.forbidden_ids,
            "limit": self.limit,
        }


@dataclass
class EvalCase:
    id: str
    description: str
    source_text: str
    app_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    prior_active: list[MemoryCandidate] = field(default_factory=list)
    candidates: list[MemoryCandidate] = field(default_factory=list)
    expected: dict[str, ExpectedCandidate] = field(default_factory=dict)
    retrieval: RetrievalExpectation | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalCase":
        expected_raw = data.get("expected", {})
        expected = {
            str(candidate_id): ExpectedCandidate.from_dict({"id": candidate_id, **expected_data})
            for candidate_id, expected_data in expected_raw.items()
        }
        return cls(
            id=str(data["id"]),
            description=str(data.get("description", "")),
            source_text=str(data.get("source_text", "")),
            app_id=data.get("app_id"),
            project_id=data.get("project_id"),
            session_id=data.get("session_id"),
            prior_active=[MemoryCandidate.from_dict(x) for x in data.get("prior_active", [])],
            candidates=[MemoryCandidate.from_dict(x) for x in data.get("candidates", [])],
            expected=expected,
            retrieval=RetrievalExpectation.from_dict(data.get("retrieval")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "source_text": self.source_text,
            "app_id": self.app_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "prior_active": [x.to_dict() for x in self.prior_active],
            "candidates": [x.to_dict() for x in self.candidates],
            "expected": {k: v.to_dict() for k, v in self.expected.items()},
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
        }


@dataclass
class ValidatedMemory:
    id: str
    status: str
    scope: str
    reasons: list[str] = field(default_factory=list)
    invalidates: list[str] = field(default_factory=list)
    confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidatedMemory":
        return cls(
            id=str(data["id"]),
            status=str(data.get("status", STATUS_REJECTED)),
            scope=str(data.get("scope", "")),
            reasons=[str(x) for x in data.get("reasons", [])],
            invalidates=[str(x) for x in data.get("invalidates", [])],
            confidence=float(data.get("confidence", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "status": self.status,
            "scope": self.scope,
            "reasons": self.reasons,
            "invalidates": self.invalidates,
            "confidence": self.confidence,
        }


@dataclass
class CaseRun:
    case_id: str
    validated: list[ValidatedMemory]
    retrieved_ids: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaseRun":
        return cls(
            case_id=str(data["case_id"]),
            validated=[ValidatedMemory.from_dict(x) for x in data.get("validated", [])],
            retrieved_ids=[str(x) for x in data.get("retrieved_ids", [])],
            timings_ms={str(k): float(v) for k, v in data.get("timings_ms", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "validated": [x.to_dict() for x in self.validated],
            "retrieved_ids": self.retrieved_ids,
            "timings_ms": self.timings_ms,
        }


@dataclass
class MemoryEvalThresholds:
    max_false_active_rate: float = 0.02
    max_secret_leak_rate: float = 0.0
    max_scope_leakage_rate: float = 0.01
    max_bad_injection_rate: float = 0.0
    max_manual_review_burden: float = 0.05
    max_contradiction_misses: int = 0
    max_validation_p95_ms: float = 50.0
    max_retrieval_p95_ms: float = 50.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_false_active_rate": self.max_false_active_rate,
            "max_secret_leak_rate": self.max_secret_leak_rate,
            "max_scope_leakage_rate": self.max_scope_leakage_rate,
            "max_bad_injection_rate": self.max_bad_injection_rate,
            "max_manual_review_burden": self.max_manual_review_burden,
            "max_contradiction_misses": self.max_contradiction_misses,
            "max_validation_p95_ms": self.max_validation_p95_ms,
            "max_retrieval_p95_ms": self.max_retrieval_p95_ms,
        }


@dataclass
class CaseEval:
    case_id: str
    status_accuracy: float
    false_active_ids: list[str] = field(default_factory=list)
    missed_active_ids: list[str] = field(default_factory=list)
    secret_leak_ids: list[str] = field(default_factory=list)
    scope_leakage_ids: list[str] = field(default_factory=list)
    contradiction_miss_ids: list[str] = field(default_factory=list)
    bad_injection_ids: list[str] = field(default_factory=list)
    retrieval_missed_ids: list[str] = field(default_factory=list)
    manual_review_ids: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status_accuracy": self.status_accuracy,
            "false_active_ids": self.false_active_ids,
            "missed_active_ids": self.missed_active_ids,
            "secret_leak_ids": self.secret_leak_ids,
            "scope_leakage_ids": self.scope_leakage_ids,
            "contradiction_miss_ids": self.contradiction_miss_ids,
            "bad_injection_ids": self.bad_injection_ids,
            "retrieval_missed_ids": self.retrieval_missed_ids,
            "manual_review_ids": self.manual_review_ids,
            "timings_ms": self.timings_ms,
        }


@dataclass
class MemoryEvalReport:
    timestamp: str
    thresholds: MemoryEvalThresholds
    summary: dict[str, Any]
    cases: list[CaseEval]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "passed": self.passed,
            "thresholds": self.thresholds.to_dict(),
            "summary": self.summary,
            "cases": [c.to_dict() for c in self.cases],
        }


class ReferenceMemoryGate:
    """Deterministic anti-garbage gate used as the suite's baseline runner."""

    def __init__(
        self,
        *,
        min_active_confidence: float = 0.72,
        min_quarantine_confidence: float = 0.55,
        min_salience: float = 0.35,
    ):
        self.min_active_confidence = min_active_confidence
        self.min_quarantine_confidence = min_quarantine_confidence
        self.min_salience = min_salience

    def run_case(self, case: EvalCase) -> CaseRun:
        start_validation = time.perf_counter()
        validated = [self.validate_candidate(case, candidate) for candidate in case.candidates]
        validation_ms = (time.perf_counter() - start_validation) * 1000

        start_retrieval = time.perf_counter()
        retrieved_ids = self.retrieve(case, validated)
        retrieval_ms = (time.perf_counter() - start_retrieval) * 1000

        return CaseRun(
            case_id=case.id,
            validated=validated,
            retrieved_ids=retrieved_ids,
            timings_ms={
                "validation": round(validation_ms, 3),
                "retrieval": round(retrieval_ms, 3),
                "total": round(validation_ms + retrieval_ms, 3),
            },
        )

    def validate_candidate(self, case: EvalCase, candidate: MemoryCandidate) -> ValidatedMemory:
        reasons: list[str] = []
        invalidates: list[str] = []

        if candidate.type not in ALLOWED_TYPES:
            return self._reject(candidate, "unsupported_type")

        if not all([candidate.id, candidate.subject, candidate.predicate, candidate.object, candidate.text, candidate.scope]):
            return self._reject(candidate, "incomplete")

        if _contains_sensitive("\n".join([candidate.text, candidate.object, candidate.source_quote or ""])):
            return self._reject(candidate, "sensitive")

        quote = (candidate.source_quote or "").strip()
        if quote and quote.lower() not in case.source_text.lower():
            return self._reject(candidate, "unsupported")
        if not quote:
            return self._reject(candidate, "missing_evidence")

        if candidate.salience < self.min_salience:
            return self._reject(candidate, "low_salience")

        if self._is_scope_leakage(case, candidate):
            return self._reject(candidate, "wrong_scope")

        if any(candidate.memory_key() == prior.memory_key() for prior in case.prior_active):
            return self._reject(candidate, "duplicate")

        if candidate.confidence < self.min_active_confidence:
            if candidate.confidence >= self.min_quarantine_confidence:
                return ValidatedMemory(
                    id=candidate.id,
                    status=STATUS_QUARANTINED,
                    scope=candidate.scope,
                    reasons=["low_confidence"],
                    confidence=candidate.confidence,
                )
            return self._reject(candidate, "low_confidence")

        supersede_text = f"{case.source_text}\n{candidate.text}".lower()
        for prior in case.prior_active:
            if self._same_slot(candidate, prior) and _norm(candidate.object) != _norm(prior.object):
                if any(signal in supersede_text for signal in SUPERSEDE_SIGNALS):
                    invalidates.append(prior.id)
                    reasons.append("supersedes_prior")
                else:
                    return ValidatedMemory(
                        id=candidate.id,
                        status=STATUS_DISPUTED,
                        scope=candidate.scope,
                        reasons=["contradiction"],
                        confidence=candidate.confidence,
                    )

        return ValidatedMemory(
            id=candidate.id,
            status=STATUS_ACTIVE,
            scope=candidate.scope,
            reasons=reasons,
            invalidates=invalidates,
            confidence=candidate.confidence,
        )

    def retrieve(self, case: EvalCase, validated: list[ValidatedMemory]) -> list[str]:
        if not case.retrieval or not case.retrieval.query.strip():
            return []

        invalidated_prior_ids = {mid for item in validated for mid in item.invalidates}
        active_by_id = {item.id: item for item in validated if item.status == STATUS_ACTIVE}
        candidate_by_id = {candidate.id: candidate for candidate in case.candidates}
        for prior in case.prior_active:
            if prior.id not in invalidated_prior_ids:
                candidate_by_id[prior.id] = prior
                active_by_id[prior.id] = ValidatedMemory(
                    id=prior.id,
                    status=STATUS_ACTIVE,
                    scope=prior.scope,
                    confidence=prior.confidence,
                )

        query_words = _words(case.retrieval.query)
        scored: list[tuple[float, str]] = []
        for memory_id, active in active_by_id.items():
            candidate = candidate_by_id.get(memory_id)
            if not candidate:
                continue
            haystack = " ".join([
                candidate.type,
                candidate.subject,
                candidate.predicate,
                candidate.object,
                candidate.text,
                candidate.scope,
            ])
            score = _overlap_score(query_words, _words(haystack))
            if score > 0:
                scored.append((score + active.confidence * 0.01, memory_id))

        scored.sort(reverse=True)
        return [memory_id for _, memory_id in scored[:case.retrieval.limit]]

    @staticmethod
    def _reject(candidate: MemoryCandidate, reason: str) -> ValidatedMemory:
        return ValidatedMemory(
            id=candidate.id,
            status=STATUS_REJECTED,
            scope=candidate.scope,
            reasons=[reason],
            confidence=candidate.confidence,
        )

    @staticmethod
    def _same_slot(a: MemoryCandidate, b: MemoryCandidate) -> bool:
        return (
            _norm(a.type),
            _norm(a.subject),
            _norm(a.predicate),
            _norm(a.scope),
        ) == (
            _norm(b.type),
            _norm(b.subject),
            _norm(b.predicate),
            _norm(b.scope),
        )

    @staticmethod
    def _is_scope_leakage(case: EvalCase, candidate: MemoryCandidate) -> bool:
        source_lower = case.source_text.lower()
        if ("this session only" in source_lower or "for this session" in source_lower) and candidate.scope != "session":
            return True
        if candidate.scope != "global":
            return False
        if case.project_id and _norm(candidate.subject) == _norm(case.project_id):
            return True
        if case.project_id and candidate.type in {"decision", "todo", "entity_note", "relationship"}:
            return True
        return False


class MemoryEvalRunner:
    """Runs the built-in or supplied memory eval cases."""

    def __init__(self, thresholds: MemoryEvalThresholds | None = None):
        self.thresholds = thresholds or MemoryEvalThresholds()

    def run(
        self,
        cases: list[EvalCase] | None = None,
        case_runs: dict[str, CaseRun] | None = None,
    ) -> MemoryEvalReport:
        cases = cases or load_builtin_cases()
        gate = ReferenceMemoryGate()
        runs: dict[str, CaseRun] = {}
        for case in cases:
            runs[case.id] = case_runs[case.id] if case_runs and case.id in case_runs else gate.run_case(case)

        case_evals = [evaluate_case(case, runs[case.id]) for case in cases]
        summary = summarize(case_evals, cases, runs)
        passed = _passes_thresholds(summary, self.thresholds)
        return MemoryEvalReport(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            thresholds=self.thresholds,
            summary=summary,
            cases=case_evals,
            passed=passed,
        )


def evaluate_case(case: EvalCase, run: CaseRun) -> CaseEval:
    expected_by_id = case.expected
    actual_by_id = {item.id: item for item in run.validated}
    all_ids = sorted(set(expected_by_id) | set(actual_by_id))

    status_matches = 0
    false_active_ids: list[str] = []
    missed_active_ids: list[str] = []
    secret_leak_ids: list[str] = []
    scope_leakage_ids: list[str] = []
    contradiction_miss_ids: list[str] = []
    manual_review_ids: list[str] = []

    for candidate_id in all_ids:
        expected = expected_by_id.get(candidate_id)
        actual = actual_by_id.get(candidate_id)
        actual_status = actual.status if actual else "missing"
        # If an engine never emits a known-bad candidate, that is acceptable: not
        # creating the garbage is equivalent to rejecting it.  Missing expected
        # active memories still count as misses below.
        effective_status = actual_status
        if actual is None and expected and expected.status != STATUS_ACTIVE:
            effective_status = expected.status

        if expected and effective_status == expected.status:
            status_matches += 1

        if actual_status == STATUS_ACTIVE and (expected is None or expected.status != STATUS_ACTIVE):
            false_active_ids.append(candidate_id)
        if expected and expected.status == STATUS_ACTIVE and actual_status != STATUS_ACTIVE:
            missed_active_ids.append(candidate_id)
        if expected and "sensitive" in expected.reasons and actual_status == STATUS_ACTIVE:
            secret_leak_ids.append(candidate_id)
        if expected and expected.scope and actual_status == STATUS_ACTIVE and actual and actual.scope != expected.scope:
            scope_leakage_ids.append(candidate_id)
        if expected and expected.invalidates:
            actual_invalidates = set(actual.invalidates if actual else [])
            missing = set(expected.invalidates) - actual_invalidates
            if missing:
                contradiction_miss_ids.append(candidate_id)
        if actual_status == STATUS_QUARANTINED:
            manual_review_ids.append(candidate_id)

    retrieval = case.retrieval
    retrieved = set(run.retrieved_ids)
    expected_retrieved = set(retrieval.expected_ids if retrieval else [])
    forbidden = set(retrieval.forbidden_ids if retrieval else [])
    non_active_retrieved = {
        memory_id
        for memory_id in retrieved
        if memory_id in expected_by_id and expected_by_id[memory_id].status != STATUS_ACTIVE
    }
    bad_injection_ids = sorted((retrieved & forbidden) | non_active_retrieved)
    retrieval_missed_ids = sorted(expected_retrieved - retrieved)

    return CaseEval(
        case_id=case.id,
        status_accuracy=round(status_matches / max(len(all_ids), 1), 4),
        false_active_ids=false_active_ids,
        missed_active_ids=missed_active_ids,
        secret_leak_ids=secret_leak_ids,
        scope_leakage_ids=scope_leakage_ids,
        contradiction_miss_ids=contradiction_miss_ids,
        bad_injection_ids=bad_injection_ids,
        retrieval_missed_ids=retrieval_missed_ids,
        manual_review_ids=manual_review_ids,
        timings_ms=run.timings_ms,
    )


def summarize(case_evals: list[CaseEval], cases: list[EvalCase], runs: dict[str, CaseRun]) -> dict[str, Any]:
    total_expected = sum(len(case.expected) for case in cases)
    total_expected_active = sum(1 for case in cases for expected in case.expected.values() if expected.status == STATUS_ACTIVE)
    total_sensitive = sum(1 for case in cases for expected in case.expected.values() if "sensitive" in expected.reasons)
    total_expected_retrieval = sum(len(case.retrieval.expected_ids) for case in cases if case.retrieval)
    total_retrieved = sum(len(run.retrieved_ids) for run in runs.values())
    actual_active = sum(1 for run in runs.values() for item in run.validated if item.status == STATUS_ACTIVE)
    actual_quarantined = sum(1 for run in runs.values() for item in run.validated if item.status == STATUS_QUARANTINED)

    false_active = _flatten(case.false_active_ids for case in case_evals)
    missed_active = _flatten(case.missed_active_ids for case in case_evals)
    secret_leaks = _flatten(case.secret_leak_ids for case in case_evals)
    scope_leaks = _flatten(case.scope_leakage_ids for case in case_evals)
    contradiction_misses = _flatten(case.contradiction_miss_ids for case in case_evals)
    bad_injections = _flatten(case.bad_injection_ids for case in case_evals)
    retrieval_misses = _flatten(case.retrieval_missed_ids for case in case_evals)

    validation_ms = [case.timings_ms.get("validation", 0.0) for case in case_evals]
    retrieval_ms = [case.timings_ms.get("retrieval", 0.0) for case in case_evals]
    total_ms = [case.timings_ms.get("total", 0.0) for case in case_evals]

    status_accuracy = sum(case.status_accuracy for case in case_evals) / max(len(case_evals), 1)
    active_recall = (total_expected_active - len(missed_active)) / max(total_expected_active, 1)
    retrieval_recall = (total_expected_retrieval - len(retrieval_misses)) / max(total_expected_retrieval, 1)

    return {
        "cases": len(cases),
        "candidates": total_expected,
        "status_accuracy": round(status_accuracy, 4),
        "active_recall": round(active_recall, 4),
        "retrieval_recall": round(retrieval_recall, 4),
        "actual_active": actual_active,
        "actual_quarantined": actual_quarantined,
        "false_active_count": len(false_active),
        "false_active_rate": round(len(false_active) / max(actual_active, 1), 4),
        "secret_leak_count": len(secret_leaks),
        "secret_leak_rate": round(len(secret_leaks) / max(total_sensitive, 1), 4),
        "scope_leakage_count": len(scope_leaks),
        "scope_leakage_rate": round(len(scope_leaks) / max(total_expected_active, 1), 4),
        "contradiction_miss_count": len(contradiction_misses),
        "bad_injection_count": len(bad_injections),
        "bad_injection_rate": round(len(bad_injections) / max(total_retrieved, 1), 4),
        "manual_review_burden": round(actual_quarantined / max(total_expected, 1), 4),
        "retrieval_missed_count": len(retrieval_misses),
        "latency_ms": {
            "validation_p50": round(_percentile(validation_ms, 50), 3),
            "validation_p95": round(_percentile(validation_ms, 95), 3),
            "retrieval_p50": round(_percentile(retrieval_ms, 50), 3),
            "retrieval_p95": round(_percentile(retrieval_ms, 95), 3),
            "total_p50": round(_percentile(total_ms, 50), 3),
            "total_p95": round(_percentile(total_ms, 95), 3),
        },
        "ids": {
            "false_active": false_active,
            "missed_active": missed_active,
            "secret_leaks": secret_leaks,
            "scope_leaks": scope_leaks,
            "contradiction_misses": contradiction_misses,
            "bad_injections": bad_injections,
            "retrieval_misses": retrieval_misses,
        },
    }


def load_builtin_cases() -> list[EvalCase]:
    return [EvalCase.from_dict(case) for case in BUILTIN_CASES]


def load_cases(path: Path | str | None = None) -> list[EvalCase]:
    if path is None:
        return load_builtin_cases()
    with open(path) as f:
        data = json.load(f)
    raw_cases = data.get("cases", data) if isinstance(data, dict) else data
    return [EvalCase.from_dict(case) for case in raw_cases]


def load_case_runs(path: Path | str) -> dict[str, CaseRun]:
    with open(path) as f:
        data = json.load(f)
    raw_runs = data.get("case_runs", data) if isinstance(data, dict) else data
    runs = [CaseRun.from_dict(run) for run in raw_runs]
    return {run.case_id: run for run in runs}


def save_report(report: MemoryEvalReport, path: Path | str) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    return out


def _passes_thresholds(summary: dict[str, Any], thresholds: MemoryEvalThresholds) -> bool:
    latency = summary["latency_ms"]
    return (
        summary["false_active_rate"] <= thresholds.max_false_active_rate
        and summary["secret_leak_rate"] <= thresholds.max_secret_leak_rate
        and summary["scope_leakage_rate"] <= thresholds.max_scope_leakage_rate
        and summary["bad_injection_rate"] <= thresholds.max_bad_injection_rate
        and summary["manual_review_burden"] <= thresholds.max_manual_review_burden
        and summary["contradiction_miss_count"] <= thresholds.max_contradiction_misses
        and latency["validation_p95"] <= thresholds.max_validation_p95_ms
        and latency["retrieval_p95"] <= thresholds.max_retrieval_p95_ms
    )


def _norm(value: str) -> str:
    return " ".join(str(value).lower().strip().split())


def _contains_sensitive(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_PATTERNS)


def _words(text: str) -> set[str]:
    out: set[str] = set()
    for raw in re.findall(r"[A-Za-z0-9_\-]+", text.lower()):
        word = raw.strip("_-")
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        if len(word) >= 3 and word not in STOPWORDS:
            out.add(word)
    return out


def _overlap_score(query_words: set[str], item_words: set[str]) -> float:
    if not query_words or not item_words:
        return 0.0
    overlap = len(query_words & item_words)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(query_words) * len(item_words))


def _flatten(groups: Any) -> list[str]:
    return [item for group in groups for item in group]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (percentile / 100)
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


BUILTIN_CASES: list[dict[str, Any]] = [
    {
        "id": "global_preference_valid",
        "description": "Durable global answer-style preference should become active.",
        "source_text": "User: For all tools and apps, keep answers short and high-signal by default.",
        "app_id": "pi",
        "candidates": [
            {
                "id": "c-pref-short",
                "type": "preference",
                "subject": "user",
                "predicate": "prefers",
                "object": "short high-signal answers",
                "text": "User prefers short, high-signal answers by default across tools.",
                "scope": "global",
                "confidence": 0.94,
                "source_quote": "keep answers short and high-signal by default",
                "salience": 0.92,
            }
        ],
        "expected": {
            "c-pref-short": {"status": "active", "scope": "global"}
        },
        "retrieval": {
            "query": "concise answer style",
            "expected_ids": ["c-pref-short"],
            "forbidden_ids": [],
        },
    },
    {
        "id": "project_decision_scope",
        "description": "Project decision should stay project-scoped; global clone is garbage.",
        "source_text": "User: In ppmlx, we decided to position it as a local temporal memory layer, not an Ollama clone.",
        "app_id": "pi",
        "project_id": "ppmlx",
        "candidates": [
            {
                "id": "c-ppmlx-position",
                "type": "decision",
                "subject": "ppmlx",
                "predicate": "positioning",
                "object": "local temporal memory layer, not an Ollama clone",
                "text": "ppmlx is positioned as a local temporal memory layer, not an Ollama clone.",
                "scope": "project",
                "confidence": 0.93,
                "source_quote": "position it as a local temporal memory layer, not an Ollama clone",
                "salience": 0.95,
            },
            {
                "id": "c-ppmlx-position-global",
                "type": "decision",
                "subject": "ppmlx",
                "predicate": "positioning",
                "object": "local temporal memory layer",
                "text": "Globally remember ppmlx positioning as local temporal memory.",
                "scope": "global",
                "confidence": 0.88,
                "source_quote": "position it as a local temporal memory layer",
                "salience": 0.8,
            },
        ],
        "expected": {
            "c-ppmlx-position": {"status": "active", "scope": "project"},
            "c-ppmlx-position-global": {"status": "rejected", "scope": "global", "reasons": ["wrong_scope"]},
        },
        "retrieval": {
            "query": "ppmlx positioning",
            "expected_ids": ["c-ppmlx-position"],
            "forbidden_ids": ["c-ppmlx-position-global"],
        },
    },
    {
        "id": "secret_rejection",
        "description": "Secrets must never become active or retrievable.",
        "source_text": "User: I pasted this by mistake: api_key=sk-test-abc123SECRET. Do not remember it.",
        "app_id": "pi",
        "candidates": [
            {
                "id": "c-secret",
                "type": "fact",
                "subject": "user",
                "predicate": "api_key",
                "object": "sk-test-abc123SECRET",
                "text": "User API key is sk-test-abc123SECRET.",
                "scope": "global",
                "confidence": 0.97,
                "source_quote": "api_key=sk-test-abc123SECRET",
                "salience": 0.9,
            }
        ],
        "expected": {
            "c-secret": {"status": "rejected", "scope": "global", "reasons": ["sensitive"]}
        },
        "retrieval": {
            "query": "api key credential",
            "expected_ids": [],
            "forbidden_ids": ["c-secret"],
        },
    },
    {
        "id": "unsupported_hallucination",
        "description": "Candidate must be backed by source evidence, not speculation.",
        "source_text": "User: Maybe next month we could explore team sync, but nothing is decided.",
        "app_id": "pi",
        "candidates": [
            {
                "id": "c-team-sync-committed",
                "type": "decision",
                "subject": "team_sync",
                "predicate": "status",
                "object": "committed for next month",
                "text": "Team sync is committed for next month.",
                "scope": "global",
                "confidence": 0.81,
                "source_quote": "team sync is committed for next month",
                "salience": 0.7,
            }
        ],
        "expected": {
            "c-team-sync-committed": {"status": "rejected", "scope": "global", "reasons": ["unsupported"]}
        },
        "retrieval": {
            "query": "team sync next month",
            "expected_ids": [],
            "forbidden_ids": ["c-team-sync-committed"],
        },
    },
    {
        "id": "low_salience_joke",
        "description": "Jokes and throwaway naming should not pollute memory.",
        "source_text": "User: Haha, call the graph a potato. Joking — do not take that seriously.",
        "app_id": "pi",
        "candidates": [
            {
                "id": "c-potato-name",
                "type": "fact",
                "subject": "temporal_context_graph",
                "predicate": "name",
                "object": "potato",
                "text": "The temporal context graph is named Potato.",
                "scope": "global",
                "confidence": 0.76,
                "source_quote": "call the graph a potato",
                "salience": 0.12,
            }
        ],
        "expected": {
            "c-potato-name": {"status": "rejected", "scope": "global", "reasons": ["low_salience"]}
        },
        "retrieval": {
            "query": "graph name",
            "expected_ids": [],
            "forbidden_ids": ["c-potato-name"],
        },
    },
    {
        "id": "temporal_supersession",
        "description": "New preference should invalidate old contradictory preference.",
        "source_text": "User: Actually, from now on I prefer concise answers, not verbose explanations.",
        "app_id": "pi",
        "prior_active": [
            {
                "id": "p-verbose",
                "type": "preference",
                "subject": "user",
                "predicate": "prefers",
                "object": "verbose explanations",
                "text": "User prefers verbose explanations.",
                "scope": "global",
                "confidence": 0.9,
                "source_quote": "prefers verbose explanations",
                "salience": 0.8,
            }
        ],
        "candidates": [
            {
                "id": "c-concise-supersedes",
                "type": "preference",
                "subject": "user",
                "predicate": "prefers",
                "object": "concise answers",
                "text": "User now prefers concise answers.",
                "scope": "global",
                "confidence": 0.95,
                "source_quote": "from now on I prefer concise answers",
                "salience": 0.9,
            }
        ],
        "expected": {
            "c-concise-supersedes": {
                "status": "active",
                "scope": "global",
                "reasons": ["supersedes_prior"],
                "invalidates": ["p-verbose"],
            }
        },
        "retrieval": {
            "query": "current answer preference concise verbose",
            "expected_ids": ["c-concise-supersedes"],
            "forbidden_ids": ["p-verbose"],
        },
    },
    {
        "id": "duplicate_prior",
        "description": "Exact duplicate of prior active memory should not create another active node.",
        "source_text": "User: Reminder: ppmlx is positioned as a local temporal memory layer.",
        "app_id": "pi",
        "project_id": "ppmlx",
        "prior_active": [
            {
                "id": "p-ppmlx-position",
                "type": "decision",
                "subject": "ppmlx",
                "predicate": "positioning",
                "object": "local temporal memory layer",
                "text": "ppmlx is positioned as a local temporal memory layer.",
                "scope": "project",
                "confidence": 0.93,
                "source_quote": "positioned as a local temporal memory layer",
                "salience": 0.9,
            }
        ],
        "candidates": [
            {
                "id": "c-duplicate-position",
                "type": "decision",
                "subject": "ppmlx",
                "predicate": "positioning",
                "object": "local temporal memory layer",
                "text": "ppmlx is positioned as a local temporal memory layer.",
                "scope": "project",
                "confidence": 0.9,
                "source_quote": "ppmlx is positioned as a local temporal memory layer",
                "salience": 0.8,
            }
        ],
        "expected": {
            "c-duplicate-position": {"status": "rejected", "scope": "project", "reasons": ["duplicate"]}
        },
        "retrieval": {
            "query": "ppmlx temporal memory positioning",
            "expected_ids": ["p-ppmlx-position"],
            "forbidden_ids": ["c-duplicate-position"],
        },
    },
    {
        "id": "session_only_scope",
        "description": "Session-only instruction should not leak into global memory.",
        "source_text": "User: For this session only, use a playful tone while brainstorming names.",
        "app_id": "pi",
        "session_id": "s-playful",
        "candidates": [
            {
                "id": "c-playful-session",
                "type": "instruction",
                "subject": "assistant",
                "predicate": "tone",
                "object": "playful while brainstorming names",
                "text": "Use a playful tone while brainstorming names in this session.",
                "scope": "session",
                "confidence": 0.9,
                "source_quote": "For this session only, use a playful tone",
                "salience": 0.75,
            },
            {
                "id": "c-playful-global",
                "type": "instruction",
                "subject": "assistant",
                "predicate": "tone",
                "object": "playful while brainstorming names",
                "text": "Always use a playful tone while brainstorming names.",
                "scope": "global",
                "confidence": 0.88,
                "source_quote": "use a playful tone while brainstorming names",
                "salience": 0.7,
            },
        ],
        "expected": {
            "c-playful-session": {"status": "active", "scope": "session"},
            "c-playful-global": {"status": "rejected", "scope": "global", "reasons": ["wrong_scope"]},
        },
        "retrieval": {
            "query": "playful tone brainstorming names session",
            "expected_ids": ["c-playful-session"],
            "forbidden_ids": ["c-playful-global"],
        },
    },
]

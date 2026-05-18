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

from ppmlx.answer_quality import AnswerQualityCase, AnswerQualityEvaluator, match_facts, select_required_facts
from ppmlx.answer_quality_replay import _post_chat, load_session_messages
from ppmlx.context_reducer import estimate_messages_tokens, group_messages_into_episodes
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
    context_missing: list[str]
    context_fact_coverage: float


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
    ablations: dict[str, Any] = field(default_factory=dict)
    probe_type: str = "answerable_text"
    classifier_reason: str = ""
    error: str | None = None
    generated_answer: str | None = None
    expected_answer: str | None = None
    required_facts: list[str] | None = None

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
            })
        return data


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
    skipped_probes: list[QualityBenchSkippedProbe] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.probes) and all(probe.passed for probe in self.probes)

    def summary(self) -> dict[str, Any]:
        ok = [probe for probe in self.probes if not probe.error]
        return {
            "probes": len(self.probes),
            "ok": len(ok),
            "passed": sum(1 for probe in ok if probe.passed),
            "failed": sum(1 for probe in ok if not probe.passed),
            "errors": sum(1 for probe in self.probes if probe.error),
            "skipped": len(self.skipped_probes),
            "skipped_by_type": _count_by_type([probe.probe_type for probe in self.skipped_probes]),
            "failure_buckets": _count_by_type([probe.failure_bucket for probe in ok]),
            "avg_recall": _avg([probe.recall for probe in ok]),
            "wrong_facts_total": sum(probe.wrong_facts for probe in ok),
            "avg_actionability": _avg([probe.actionability for probe in ok]),
            "avg_grounding": _avg([probe.grounding for probe in ok]),
            "avg_equivalence": _avg([probe.equivalence for probe in ok]),
            "avg_context_fact_coverage": _avg([probe.context_fact_coverage for probe in ok]),
            "avg_prompt_compression": _avg([probe.prompt_compression for probe in ok]),
            "avg_replay_compression": _avg([probe.replay_compression for probe in ok]),
            "prompt_tokens_total": sum(probe.prompt_tokens for probe in ok),
            "completion_tokens_total": sum(probe.completion_tokens for probe in ok),
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
            "summary": self.summary(),
            "probes": [probe.to_dict(include_content=include_content) for probe in self.probes],
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
    responder: Responder | None = None,
) -> QualityBenchReport:
    resolved_source, messages = load_session_messages(path, source=source)
    probes, skipped = build_quality_probes(
        messages,
        split=split,
        max_probes=max_probes,
        include_probe_types=include_probe_types,
    )
    results: list[QualityBenchProbeResult] = []
    all_skipped = list(skipped)
    for probe in probes:
        prepared = prepare_quality_probe(probe, model=model)
        if not prepared.required_facts:
            all_skipped.append(QualityBenchSkippedProbe(
                probe_id=probe.probe_id,
                episode_index=probe.episode_index,
                probe_type="oracle_unavailable_in_context",
                reason="no expected-answer oracle facts are recoverable from compact/replay context",
            ))
            continue
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
        skipped_probes=all_skipped,
    )


def prepare_quality_probe(probe: QualityBenchProbe, *, model: str) -> QualityBenchPreparedProbe:
    """Prepare replay/oracle facts, keeping only facts visible in compact context.

    The recorded next assistant answer often contains new synthesis or tool-loop
    actions that are not present in the prefix. Those are useful diagnostics but
    not fair recall requirements for a text-only compact-context probe.
    """
    messages = [*probe.prefix_messages, probe.user_message]
    replay = _probe_replay(messages, model=model, probe_id=probe.probe_id)
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
    context_found, context_missing = match_facts(source_context, raw_required)
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
        context_missing=context_missing,
        context_fact_coverage=_ratio(len(context_found), len(raw_required)),
    )


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
) -> QualityBenchProbeResult:
    messages = [_quality_probe_system_message(), *probe.prefix_messages, probe.user_message]
    metadata = {"app_id": "quality-bench", "project_id": f"quality-bench-{source}", "session_id": probe.probe_id}
    started = time.time()
    try:
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
        prepared = prepared or prepare_quality_probe(probe, model=model)
        replay = prepared.replay
        source_context = prepared.source_context
        required_facts = prepared.required_facts
        context_missing = prepared.context_missing
        context_fact_coverage = prepared.context_fact_coverage
        case = AnswerQualityCase(
            case_id=probe.probe_id,
            question=_message_text(probe.user_message),
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
            expected_actions=["next"],
        )
        evaluator = AnswerQualityEvaluator()
        quality = evaluator.evaluate_case(case)
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
            ablations=_ablation_diagnostics(replay),
            probe_type=probe.probe_type,
            classifier_reason=probe.classifier_reason,
            generated_answer=generated if include_content else None,
            expected_answer=probe.expected_answer if include_content else None,
            required_facts=required_facts if include_content else None,
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


def _quality_probe_system_message() -> dict[str, str]:
    return {
        "role": "system",
        "content": (
            "You are answering a memory quality benchmark probe. Use only facts visible in the prior "
            "conversation or recovered memory context. Be concise and concrete. Preserve exact identifiers, "
            "commands, filenames, model names, numbers, and decisions when relevant. Do not speculate, add "
            "generic caveats, or claim you cannot access earlier context."
        ),
    }


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


def _probe_replay(messages: list[dict[str, Any]], *, model: str, probe_id: str) -> dict[str, Any]:
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
    return compact_replay(trace).to_dict()


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
    if any(marker in user_norm for marker in phrase_markers):
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
    )
    user_tokens = set(user_norm.replace("-", " ").split())
    return bool(user_tokens & set(action_verbs)) and (any(term in user_tokens for term in repo_terms) or expected_action)


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

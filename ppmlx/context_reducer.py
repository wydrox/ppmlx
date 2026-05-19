"""Rolling context reducer backed by the local temporal memory graph.

Compact mode keeps only a small hot tail in the prompt and renders a lean session
context from memory graph items.  Older closed messages are written into the
memory engine before inference so long OpenAI-compatible histories do not have
to be sent wholesale to local models.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any

from ppmlx.memory_engine import get_memory_engine
from ppmlx.memory_store import get_memory_store


_CONTEXT_ROLE = "system"
_CONTEXT_TITLE = "Compacted local session context"


@dataclass
class ContextBudget:
    rolling_tokens: int = 10000
    hot_tail_tokens: int = 6500
    session_context_tokens: int = 2000
    compact_threshold_tokens: int = 12000
    max_context_items: int = 40
    mode: str = "off"
    extract_cold_messages: bool = False

    @classmethod
    def from_config(cls) -> "ContextBudget":
        try:
            from ppmlx.config import load_config

            memory = load_config().memory
            mode = "off" if not bool(getattr(memory, "enabled", True)) else str(getattr(memory, "mode", "off")).lower()
            return cls(
                rolling_tokens=int(getattr(memory, "rolling_tokens", cls.rolling_tokens)),
                hot_tail_tokens=int(getattr(memory, "hot_tail_tokens", cls.hot_tail_tokens)),
                session_context_tokens=int(getattr(memory, "session_context_tokens", cls.session_context_tokens)),
                compact_threshold_tokens=int(getattr(memory, "compact_threshold_tokens", cls.compact_threshold_tokens)),
                max_context_items=int(getattr(memory, "max_context_items", cls.max_context_items)),
                mode=mode,
            )
        except Exception:
            return cls()


@dataclass
class Episode:
    index: int
    messages: list[dict[str, Any]]

    @property
    def tokens(self) -> int:
        return estimate_messages_tokens(self.messages)


@dataclass
class HandoffResult:
    context: str
    items: list[dict[str, Any]]
    tokens: int
    query: str | None
    app_id: str | None
    project_id: str | None
    session_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context,
            "tokens": self.tokens,
            "items_count": len(self.items),
            "query": self.query,
            "app_id": self.app_id,
            "project_id": self.project_id,
            "session_id": self.session_id,
            "items": self.items,
        }


@dataclass
class ReductionResult:
    messages: list[dict[str, Any]]
    original_tokens: int
    reduced_tokens: int
    hot_tail_tokens: int
    session_context_tokens: int
    cold_messages: int
    context_items: int
    compacted: bool
    injected: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return self.compacted or self.injected

    def to_metadata(self) -> dict[str, Any]:
        return {
            "original_tokens": self.original_tokens,
            "reduced_tokens": self.reduced_tokens,
            "hot_tail_tokens": self.hot_tail_tokens,
            "session_context_tokens": self.session_context_tokens,
            "cold_messages": self.cold_messages,
            "context_items": self.context_items,
            "compacted": self.compacted,
            "injected": self.injected,
            **self.metadata,
        }


class ContextReducer:
    """Reduce long chat histories to a hot tail plus scoped graph context."""

    def __init__(self, budget: ContextBudget | None = None, store=None, engine=None):
        self.budget = budget or ContextBudget.from_config()
        self.store = store
        self.engine = engine

    def reduce(
        self,
        *,
        request_id: str,
        model_alias: str,
        model_repo: str,
        messages: list[dict[str, Any]],
        memory_context: dict | None = None,
    ) -> ReductionResult:
        original_tokens = estimate_messages_tokens(messages)
        mode = self.budget.mode
        if mode not in {"compact", "inject"}:
            return ReductionResult(
                messages=messages,
                original_tokens=original_tokens,
                reduced_tokens=original_tokens,
                hot_tail_tokens=original_tokens,
                session_context_tokens=0,
                cold_messages=0,
                context_items=0,
                compacted=False,
                injected=False,
            )

        system_messages, non_system = _split_system(messages)
        should_compact = original_tokens > self.budget.compact_threshold_tokens
        if should_compact:
            hot_tail, cold = self._select_hot_tail(non_system)
        else:
            hot_tail, cold = non_system, []

        context_info = memory_context or {}
        if cold and self.budget.extract_cold_messages:
            self._compact_cold_messages(
                request_id=request_id,
                model_alias=model_alias,
                model_repo=model_repo,
                cold_messages=cold,
                memory_context=context_info,
            )

        retrieval_start = time.perf_counter()
        context_items = self._retrieve_context_items(hot_tail, context_info)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000
        context_text = render_session_context(
            context_items,
            max_tokens=self.budget.session_context_tokens,
            store=self.store or get_memory_store(),
        )
        context_message = {"role": _CONTEXT_ROLE, "content": context_text} if context_text else None
        reduced_messages = [*system_messages]
        if context_message:
            reduced_messages.append(context_message)
        reduced_messages.extend(hot_tail)

        reduced_tokens = estimate_messages_tokens(reduced_messages)
        return ReductionResult(
            messages=reduced_messages,
            original_tokens=original_tokens,
            reduced_tokens=reduced_tokens,
            hot_tail_tokens=estimate_messages_tokens(hot_tail),
            session_context_tokens=estimate_text_tokens(context_text),
            cold_messages=len(cold),
            context_items=len(context_items),
            compacted=bool(cold),
            injected=bool(context_message),
            metadata={
                "mode": mode,
                "system_messages": len(system_messages),
                "retrieval_latency_ms": round(retrieval_latency_ms, 3),
            },
        )

    def _select_hot_tail(self, non_system: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not non_system:
            return [], []
        episodes = group_messages_into_episodes(non_system)
        selected_reversed: list[Episode] = []
        total = 0
        for idx, episode in enumerate(reversed(episodes)):
            if idx == 0 and episode.tokens > self.budget.hot_tail_tokens:
                # Real agent traces can contain a single huge user→assistant→tool
                # episode with dozens of tool calls. Keeping it whole defeats
                # compaction, so split only the oversized newest episode by
                # message tail while preserving the most recent local chain.
                hot, cold = _split_oversized_episode_tail(episode.messages, self.budget.hot_tail_tokens)
                cold_episodes = episodes[: len(episodes) - 1]
                cold_messages = [*_flatten_messages(cold_episodes), *cold]
                return hot, cold_messages
            if idx > 0 and total + episode.tokens > self.budget.hot_tail_tokens:
                remaining_budget = max(0, self.budget.hot_tail_tokens - total)
                if episode.tokens > self.budget.hot_tail_tokens * 2 and remaining_budget >= 120:
                    split_hot, split_cold = _split_oversized_episode_tail(episode.messages, remaining_budget)
                    cold_episodes = episodes[: len(episodes) - idx - 1]
                    hot_messages = [*split_hot, *_flatten_messages(list(reversed(selected_reversed)))]
                    cold_messages = [*_flatten_messages(cold_episodes), *split_cold]
                    return hot_messages, cold_messages
                break
            selected_reversed.append(episode)
            total += episode.tokens
        hot_episodes = list(reversed(selected_reversed))
        cold_episodes = episodes[: len(episodes) - len(hot_episodes)]
        return _flatten_messages(hot_episodes), _flatten_messages(cold_episodes)

    def _compact_cold_messages(
        self,
        *,
        request_id: str,
        model_alias: str,
        model_repo: str,
        cold_messages: list[dict[str, Any]],
        memory_context: dict,
    ) -> None:
        if not cold_messages:
            return
        engine = self.engine or get_memory_engine()
        for episode in group_messages_into_episodes(cold_messages):
            digest = sha1(json.dumps(episode.messages, sort_keys=True, default=str).encode()).hexdigest()[:12]
            event_id = f"{request_id}-compact-e{episode.index}-{digest}"
            engine.capture_chat(
                request_id=event_id,
                endpoint="/v1/chat/completions#compact",
                model_alias=model_alias,
                model_repo=model_repo,
                messages=episode.messages,
                response_text=None,
                app_id=memory_context.get("app_id"),
                project_id=memory_context.get("project_id"),
                session_id=memory_context.get("session_id"),
                metadata={
                    **(memory_context.get("metadata") or {}),
                    "compaction_parent_request_id": request_id,
                    "episode_index": episode.index,
                    "episode_messages": len(episode.messages),
                    "episode_tokens_estimate": episode.tokens,
                    "cold_messages_total": len(cold_messages),
                    "cold_tokens_estimate": estimate_messages_tokens(cold_messages),
                },
            )

    def _retrieve_context_items(self, hot_tail: list[dict[str, Any]], memory_context: dict) -> list[dict[str, Any]]:
        store = self.store or get_memory_store()
        project_id = memory_context.get("project_id")
        query = build_retrieval_query(hot_tail, store=store, project_id=project_id)
        intent_query = build_current_intent_query(hot_tail)
        scoped = dict(
            app_id=memory_context.get("app_id"),
            project_id=project_id,
            session_id=memory_context.get("session_id"),
        )
        rows: list[dict[str, Any]] = []
        fetch_limit = max(self.budget.max_context_items * 4, self.budget.max_context_items + 20)
        workflow_intent = is_generic_workflow_action_query(intent_query)
        if query and not workflow_intent:
            rows.extend(store.search(query, status="active", limit=fetch_limit, **scoped))
            rows = _filter_relevant_context_rows(rows, intent_query)
        if len(rows) < fetch_limit:
            fallback = store.query_candidates(
                status="active",
                limit=fetch_limit,
                **scoped,
            )
            if query and not workflow_intent:
                fallback = _filter_relevant_context_rows(fallback, intent_query)
            rows.extend(fallback)
        
        # Embedding re-rank: when an embedding engine is available, re-rank the
        # top results by semantic similarity to the intent query.  This catches
        # matches like "dense chunker" ↔ "sliding window embedder" that FTS5 misses.
        if rows and intent_query and not workflow_intent:
            rows = _embedding_rerank(rows, intent_query, top_k=self.budget.max_context_items * 2)
        
        return _curate_context_rows(
            rows,
            query=intent_query or query,
            scoped=scoped,
            workflow_intent=workflow_intent,
        )[: self.budget.max_context_items]


def reduce_chat_context(
    *,
    request_id: str,
    model_alias: str,
    model_repo: str,
    messages: list[dict[str, Any]],
    memory_context: dict | None = None,
) -> ReductionResult:
    return ContextReducer().reduce(
        request_id=request_id,
        model_alias=model_alias,
        model_repo=model_repo,
        messages=messages,
        memory_context=memory_context,
    )


def build_handoff_context(
    *,
    query: str | None = None,
    app_id: str | None = None,
    project_id: str | None = None,
    session_id: str | None = None,
    max_items: int = 40,
    max_tokens: int = 2000,
    store=None,
) -> HandoffResult:
    """Render the compacted session context for a live namespace.

    This is the debug/inspection path for compact mode: it shows the exact style
    of system context that can be injected before a hot tail, without needing to
    send a request through the model.
    """
    memory_store = store or get_memory_store()
    scoped = {
        "app_id": app_id,
        "project_id": project_id,
        "session_id": session_id,
    }
    rows: list[dict[str, Any]] = []
    fetch_limit = max(max_items * 4, max_items + 20)
    if query:
        rows.extend(memory_store.search(query, status="active", limit=fetch_limit, **scoped))
    if len(rows) < fetch_limit:
        rows.extend(memory_store.query_candidates(status="active", limit=fetch_limit, **scoped))
    items = _curate_context_rows(
        rows,
        query=query or "",
        scoped=scoped,
        workflow_intent=is_generic_workflow_action_query(query or ""),
    )[:max_items]
    context = render_session_context(items, max_tokens=max_tokens, store=store)
    return HandoffResult(
        context=context,
        items=items,
        tokens=estimate_text_tokens(context),
        query=query,
        app_id=app_id,
        project_id=project_id,
        session_id=session_id,
    )


def render_session_context(items: list[dict[str, Any]], *, max_tokens: int, store=None) -> str:
    if not items or max_tokens <= 0:
        return ""
    
    # Enrich with inferred edges: for each retrieved candidate, check if it
    # has inferred connections.  These are tagged [inferred] so the model knows
    # they haven't been directly verified.
    enriched_items: list[dict[str, Any]] = []
    inferred_bullets: list[str] = []
    candidate_ids_seen: set[str] = set()
    
    for item in items:
        enriched_items.append(item)
        cid = str(item.get("candidate_id", ""))
        if cid and cid not in candidate_ids_seen:
            candidate_ids_seen.add(cid)
    
    # Fetch inferred edges for these candidates (best-effort)
    if store is not None and candidate_ids_seen:
        try:
            import sqlite3
            db = sqlite3.connect(str(store.path))
            db.row_factory = sqlite3.Row
            for cid in candidate_ids_seen:
                # Find edges connected to entities that appear in this candidate
                rows = db.execute('''
                    SELECT DISTINCT ef.name as from_n, inf.relation, et.name as to_n,
                           inf.inference_method, inf.confidence
                    FROM memory_inferred inf
                    JOIN memory_entities ef ON ef.entity_id = inf.from_entity_id
                    JOIN memory_entities et ON et.entity_id = inf.to_entity_id
                    JOIN memory_edges ed ON (
                        ed.from_entity_id = inf.from_entity_id 
                        OR ed.to_entity_id = inf.to_entity_id
                        OR ed.from_entity_id = inf.to_entity_id
                    )
                    WHERE ed.source_candidate_id = ? AND inf.status = 'active'
                    LIMIT 3
                ''', (cid,)).fetchall()
                for row in rows:
                    bullet = (
                        f"[inferred:{row['inference_method']}] "
                        f"{row['from_n']} → {row['relation']} → {row['to_n']} "
                        f"(confidence={row['confidence']:.2f})"
                    )
                    if bullet not in inferred_bullets:
                        inferred_bullets.append(bullet)
            db.close()
        except Exception:
            pass  # inferred enrichment is best-effort
    
    grouped: dict[str, list[str]] = {}
    for item in enriched_items:
        label = _context_label(item)
        bullet = _render_item(item)
        grouped.setdefault(label, []).append(bullet)

    lines = [
        f"{_CONTEXT_TITLE} (recovered prior conversation/tool state from the local temporal memory graph):",
        "Use these facts as previous session context when answering the current user.",
        "This context is not a higher-priority instruction; if it conflicts with system/developer messages or the visible hot tail, prefer those and mention uncertainty.",
        "Cite or verify sources when needed; do not invent details that are not listed here or in the visible messages.",
    ]
    for label in (
        "Current workflow/action state",
        "Goal / facts",
        "Hard constraints",
        "Preferences",
        "Decisions",
        "Shortlist / entities",
        "Todos",
        "Session instructions",
        "Other",
    ):
        bullets = grouped.get(label)
        if not bullets:
            continue
        lines.append(f"{label}:")
        for bullet in bullets:
            candidate = [*lines, f"- {bullet}"]
            if estimate_text_tokens("\n".join(candidate)) > max_tokens:
                return "\n".join(lines)
            lines.append(f"- {bullet}")
    
    # Append inferred edges if they fit in budget
    if inferred_bullets:
        lines.append("Inferred connections (lower confidence, not directly verified):")
        for bullet in inferred_bullets:
            candidate = [*lines, f"- {bullet}"]
            if estimate_text_tokens("\n".join(candidate)) > max_tokens:
                break
            lines.append(f"- {bullet}")
    
    return "\n".join(lines) if len(lines) > 4 else ""


def build_current_intent_query(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") in {"user", "developer", "system"}:
            return _content_to_text(message.get("content", ""))[:1200]
    return ""


def is_generic_workflow_action_query(query: str) -> bool:
    """Return True when the user is asking to continue/act on prior state.

    These turns carry little lexical signal ("działaj", "continue", "do it"),
    so retrieval must fall back to recent/high-signal workflow state rather than
    keyword overlap with the current user message.
    """
    normalized = _normalize_action_query(query)
    if not normalized:
        return False
    phrases = {
        "dzialaj",
        "działaj",
        "kontynuuj",
        "dalej",
        "zrob to",
        "zrób to",
        "zrob to za mnie",
        "zrób to za mnie",
        "rob dalej",
        "rób dalej",
        "jedziemy",
        "continue",
        "continue please",
        "go ahead",
        "proceed",
        "do it",
        "do this",
        "carry on",
        "keep going",
    }
    if normalized in phrases:
        return True
    tokens = normalized.split()
    if len(tokens) > 5:
        return False
    generic = {
        "dzialaj", "działaj", "kontynuuj", "dalej", "zrob", "zrób", "rob", "rób", "to", "tym",
        "continue", "please", "go", "ahead", "proceed", "do", "this", "it", "carry", "on", "keep", "going",
    }
    return bool(tokens) and all(token in generic for token in tokens)


def build_retrieval_query(
    messages: list[dict[str, Any]],
    *,
    store=None,
    project_id: str | None = None,
) -> str:
    """Build retrieval query from user messages, expanded with recent active facts.
    
    Query expansion makes generic queries like "działaj" or "continue" match
    the current workflow state by appending terms from the 5 most recent active
    facts in this project.
    """
    parts: list[str] = []
    for message in reversed(messages[-8:]):
        role = message.get("role")
        if role in {"user", "developer", "system"}:
            parts.append(_content_to_text(message.get("content", "")))
        if estimate_text_tokens("\n".join(parts)) >= 600:
            break
    
    # Query expansion: append recent fact text for better lexical overlap.
    if store is not None and project_id:
        try:
            recent = store.query_candidates(
                status="active", project_id=project_id, limit=5,
            )
            fact_terms = " ".join(
                f"{c.get('subject','')} {c.get('predicate','')} {c.get('object','')}"
                for c in recent
            )
            if fact_terms.strip():
                parts.append(fact_terms)
        except Exception:
            pass  # query expansion is best-effort
    
    return "\n".join(reversed(parts))[:3000]


def group_messages_into_episodes(messages: list[dict[str, Any]]) -> list[Episode]:
    """Group non-system messages into closed task episodes.

    An episode starts at a user/developer turn and includes following assistant,
    tool, and MCP/tool-result messages until the next user/developer turn. This
    avoids compacting half of a tool interaction into the graph while keeping the
    other half in the hot prompt tail.
    """
    episodes: list[Episode] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", ""))
        starts_episode = role in {"user", "developer"}
        if starts_episode and current:
            episodes.append(Episode(index=len(episodes), messages=current))
            current = []
        current.append(message)
    if current:
        episodes.append(Episode(index=len(episodes), messages=current))
    return episodes


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(estimate_message_tokens(message) for message in messages)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    # JSON keeps role/tool metadata in the estimate and works for multipart messages.
    try:
        raw = json.dumps(message, ensure_ascii=False, default=str)
    except TypeError:
        raw = str(message)
    return estimate_text_tokens(raw) + 4


def estimate_text_tokens(text: str | None) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def _split_system(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_messages: list[dict[str, Any]] = []
    non_system: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            system_messages.append(message)
        else:
            non_system.append(message)
    return system_messages, non_system


def _flatten_messages(episodes: list[Episode]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for episode in episodes:
        out.extend(episode.messages)
    return out


def _split_oversized_episode_tail(messages: list[dict[str, Any]], token_budget: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not messages:
        return [], []
    selected_reversed: list[dict[str, Any]] = []
    total = 0
    for message in reversed(messages):
        tokens = estimate_message_tokens(message)
        if selected_reversed and total + tokens > token_budget:
            break
        if not selected_reversed and tokens > token_budget:
            hot_message, cold_message = _split_oversized_message_tail(message, token_budget)
            cold = messages[: len(messages) - 1]
            if cold_message is not None:
                cold.append(cold_message)
            return [hot_message], cold
        selected_reversed.append(message)
        total += tokens
    hot = list(reversed(selected_reversed))
    cold = messages[: len(messages) - len(hot)]
    return hot, cold


def _split_oversized_message_tail(message: dict[str, Any], token_budget: int) -> tuple[dict[str, Any], dict[str, Any] | None]:
    content = message.get("content", "")
    if not isinstance(content, str) or token_budget <= 24:
        return message, None
    max_chars = max(64, (token_budget - 16) * 4)
    if len(content) <= max_chars:
        return message, None
    split_at = max(0, len(content) - max_chars)
    while split_at > 0 and split_at < len(content) and not content[split_at].isspace():
        split_at += 1
        if len(content) - split_at < 64:
            split_at = max(0, len(content) - max_chars)
            break
    head = content[:split_at].rstrip()
    tail = content[split_at:].lstrip()
    hot = dict(message)
    hot["content"] = tail
    hot["metadata"] = {
        **(message.get("metadata") if isinstance(message.get("metadata"), dict) else {}),
        "context_tail_trim": {"part": "tail", "omitted_chars": len(head)},
    }
    cold = dict(message)
    cold["content"] = head
    cold["metadata"] = {
        **(message.get("metadata") if isinstance(message.get("metadata"), dict) else {}),
        "context_tail_trim": {"part": "head", "kept_tail_chars": len(tail)},
    }
    return hot, cold if head else None


def _filter_relevant_context_rows(rows: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_terms = set(_relevance_terms(query))
    if not query_terms or not _should_filter_fallback(query_terms):
        return rows
    relevant: list[dict[str, Any]] = []
    for row in rows:
        haystack = " ".join(str(row.get(key) or "") for key in ("text", "subject", "predicate", "object", "type"))
        row_terms = set(_relevance_terms(haystack))
        overlap = query_terms & row_terms
        if overlap or _is_high_signal_context_row(row, query_terms):
            relevant.append(row)
    return relevant


def _should_filter_fallback(query_terms: set[str]) -> bool:
    # Keep normal product/planning eval recall broad. Use stricter fallback only
    # when the user asks about a specific project/runtime identity, which is
    # where embedded fixture facts most often pollute real session handoffs.
    return bool(query_terms & {"ppmlx", "mempalace", "devryn"})


def _is_high_signal_context_row(row: dict[str, Any], query_terms: set[str]) -> bool:
    if not query_terms:
        return True
    row_type = str(row.get("type") or "").lower()
    if row_type not in {"decision", "todo", "constraint", "fact", "preference"}:
        return False
    # Keep high-signal project state only when it shares at least one non-generic
    # topical term with the query. This prevents unrelated embedded fixtures from
    # filling handoff context during real-session quality checks.
    haystack = " ".join(str(row.get(key) or "") for key in ("text", "subject", "object"))
    row_terms = set(_relevance_terms(haystack))
    return bool(query_terms & row_terms)


def _relevance_terms(text: str) -> list[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "current",
        "session", "context", "handoff", "answer", "brief", "concise", "factual",
        "state", "status", "important", "decision", "decisions", "next", "action",
        "task", "user", "assistant", "goal", "goals", "todo", "validation",
    }
    return [term for term in re.findall(r"[a-z0-9_]+", text.lower()) if len(term) >= 3 and term not in stop]


def _curate_context_rows(
    rows: list[dict[str, Any]],
    *,
    query: str,
    scoped: dict[str, Any],
    workflow_intent: bool = False,
) -> list[dict[str, Any]]:
    """Filter unsafe default retrieval and apply deterministic context ranking."""
    visible_rows = [
        row for row in _dedupe_rows(rows)
        if not is_noisy_context_namespace(row) or _has_explicit_matching_namespace(row, scoped)
    ]
    return sorted(
        visible_rows,
        key=lambda row: _context_row_rank(row, query, workflow_intent=workflow_intent),
        reverse=True,
    )


def is_noisy_context_namespace(row: dict[str, Any]) -> bool:
    """Return True for eval/test/internal namespaces hidden from general recall.

    The reducer may retrieve globally when no app/project/session filter is supplied.
    In that mode, local quality benches, answer-quality dogfood, eval runs, and
    test traces are too easy to leak into unrelated compact/handoff context.  Keep
    detection limited to namespace fields so synthetic eval content in ordinary
    project fixtures remains retrievable.
    """
    namespace = " ".join(str(row.get(key) or "") for key in ("app_id", "project_id", "session_id")).lower()
    if not namespace:
        return False
    normalized = re.sub(r"[^a-z0-9]+", "-", namespace).strip("-")
    noisy_phrases = (
        "quality-bench",
        "answer-quality-real",
        "answer-quality-eval",
        "dogfood",
        "eval",
        "test",
    )
    parts = set(normalized.split("-"))
    return any(phrase in normalized for phrase in noisy_phrases[:4]) or bool(parts & {"eval", "test"})


def _has_explicit_matching_namespace(row: dict[str, Any], scoped: dict[str, Any]) -> bool:
    for key in ("app_id", "project_id", "session_id"):
        requested = scoped.get(key)
        if requested and str(row.get(key) or "") == str(requested):
            return True
    return False


def _context_row_rank(row: dict[str, Any], query: str, *, workflow_intent: bool = False) -> tuple[int, int, int, float, float, str, str]:
    relevance = _context_relevance_score(row, query)
    type_boost = _context_type_boost(row)
    workflow_boost = _workflow_row_boost(row) if workflow_intent else 0
    return (
        workflow_boost,
        relevance,
        type_boost,
        _safe_float(row.get("salience")),
        _safe_float(row.get("confidence")),
        str(row.get("created_at") or ""),
        str(row.get("candidate_id") or ""),
    )


def _context_type_boost(row: dict[str, Any]) -> int:
    row_type = str(row.get("type") or "").lower()
    predicate = str(row.get("predicate") or "").lower()
    text = " ".join(str(row.get(key) or "") for key in ("text", "predicate", "object")).lower()
    if row_type == "workflow_state" and predicate in {"current_task", "next_action", "blocker", "command_run"}:
        return 11
    if predicate in {"commit", "commit_pushed"} or re.search(r"\b[0-9a-f]{7,40}\b", text):
        return 10
    if predicate in {"global_fix", "auth_race_fix"} or any(term in text for term in ("global fix", "auth-race", "convexproviderwithauth")):
        return 9
    if predicate == "validation" or any(term in text for term in ("pnpm build", "eslint", "pytest", "ruff", "origin/dev", "✅")):
        return 8
    if predicate == "file_changed":
        return 7
    if row_type == "decision":
        return 6
    if row_type == "constraint":
        return 5
    if row_type == "todo":
        return 4
    if row_type == "workflow_state":
        return 3
    return 0


def _context_relevance_score(row: dict[str, Any], query: str) -> int:
    query_terms = set(_relevance_terms(query))
    if not query_terms:
        return 0
    haystack = " ".join(str(row.get(key) or "") for key in ("text", "subject", "predicate", "object", "type"))
    row_terms = set(_relevance_terms(haystack))
    return len(query_terms & row_terms)


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        out.append(row)
    return out


def _render_item(item: dict[str, Any]) -> str:
    subject = str(item.get("subject") or "").strip()
    predicate = str(item.get("predicate") or "").strip()
    object_ = str(item.get("object") or "").strip()
    text = str(item.get("text") or "").strip()
    scope = str(item.get("scope") or "").strip()
    confidence = item.get("confidence")
    source = item.get("source_quote")
    core = text or " ".join(part for part in [subject, predicate, object_] if part)
    if source:
        core += f" [source: {str(source)[:120]}]"
    suffix = []
    if scope:
        suffix.append(f"scope={scope}")
    if confidence is not None:
        try:
            suffix.append(f"confidence={float(confidence):.2f}")
        except (TypeError, ValueError):
            pass
    if suffix:
        core += " (" + ", ".join(suffix) + ")"
    return core


def _context_label(item: dict[str, Any]) -> str:
    if _workflow_row_boost(item) > 0:
        return "Current workflow/action state"
    return _type_label(str(item.get("type", "memory")))


def _workflow_row_boost(row: dict[str, Any]) -> int:
    row_type = str(row.get("type") or "").lower()
    predicate = str(row.get("predicate") or "").lower()
    text = " ".join(str(row.get(key) or "") for key in ("text", "predicate", "object")).lower()
    if row_type == "workflow_state" and predicate in {"current_task", "next_action", "blocker"}:
        return 100
    if predicate in {"commit", "commit_pushed", "validation", "file_changed", "command_run"}:
        return 90
    if predicate in {"global_fix", "auth_race_fix"}:
        return 85
    if row_type == "todo" or predicate in {"needs", "todo", "next_action"}:
        return 80
    if row_type == "decision" and any(marker in text for marker in ("fix", "changed", "validation", "commit", "next", "todo", "block")):
        return 70
    if row_type == "workflow_state":
        return 60
    return 0


def _type_label(type_: str) -> str:
    normalized = type_.lower()
    if normalized == "workflow_state":
        return "Current workflow/action state"
    if normalized == "preference":
        return "Preferences"
    if normalized == "decision":
        return "Decisions"
    if normalized == "todo":
        return "Todos"
    if normalized == "instruction":
        return "Session instructions"
    if normalized in {"entity_note", "relationship"}:
        return "Shortlist / entities"
    if normalized == "fact":
        return "Goal / facts"
    if normalized == "constraint":
        return "Hard constraints"
    return "Other"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
        return " ".join(parts)
    return str(content)


def _embedding_rerank(
    rows: list[dict[str, Any]],
    query: str,
    *,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    """Re-rank candidates by embedding cosine similarity to the query.
    
    Only runs when there are more candidates than top_k AND an embedding
    engine is available.  Falls back to the original order otherwise.
    Best-effort, never throws.
    """
    if len(rows) <= top_k:
        return rows  # nothing to re-rank
    
    # Try cached embeddings first (stored by contrastive retriever)
    cached = _load_cached_embeddings(rows)
    if cached is not None and len(cached) >= len(rows) * 0.8:
        return _rerank_with_cache(rows, query, cached, top_k)
    
    # Fallback: truncate to top_k by original rank (FTS5 BM25)
    return rows[:top_k]


def _load_cached_embeddings(rows: list[dict[str, Any]]) -> dict[str, list[float]] | None:
    """Try to load pre-computed embeddings from the entity_alias cache."""
    try:
        from ppmlx.memory_store import get_memory_store
        store = get_memory_store()
        candidate_ids = [r.get("candidate_id", "") for r in rows if r.get("candidate_id")]
        if not candidate_ids:
            return None
        # Query embedding_cache aliases
        aliases = store.query_entity_aliases(
            type="embedding_cache", scope="system", active_only=True, limit=10000,
        )
        result: dict[str, list[float]] = {}
        import json
        for alias in aliases:
            meta = alias.get("metadata", {})
            if isinstance(meta, str):
                meta = json.loads(meta)
            cid = meta.get("candidate_id", "")
            vec = meta.get("vector", [])
            if cid and cid in candidate_ids and vec:
                result[cid] = vec
        return result if result else None
    except Exception:
        return None


def _rerank_with_cache(
    rows: list[dict[str, Any]],
    query: str,
    cached: dict[str, list[float]],
    top_k: int,
) -> list[dict[str, Any]]:
    """Re-rank using cached embeddings + query embedding."""
    try:
        import numpy as np
        from ppmlx.engine_embed import get_embed_engine
        
        # Only embed the query (1 call, ~30ms)
        embed_engine = get_embed_engine()
        q_vecs = embed_engine.encode(
            "qwen3-embedding:0.6b-4bit-dwq", [query[:200]], normalize=True,
        )
        query_vec = np.array(q_vecs[0], dtype=np.float32)
        
        # Score each row with cached vector
        scored: list[tuple[float, dict]] = []
        for r in rows:
            cid = r.get("candidate_id", "")
            vec = cached.get(cid)
            if vec is not None:
                sim = float(np.dot(query_vec, np.array(vec, dtype=np.float32)))
                scored.append((sim, r))
        
        if not scored:
            return rows[:top_k]
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]
    except Exception:
        return rows[:top_k]


def _normalize_action_query(query: str) -> str:
    text = str(query or "").lower()
    text = re.sub(r"[`*_>\[\]{}()!?.,:;]+", " ", text)
    return " ".join(text.split())

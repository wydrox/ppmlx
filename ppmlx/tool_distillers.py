"""Tool/MCP output distillers for temporal memory extraction.

Distillers turn structured tool results into small evidence-backed candidate
records. They do not write to the graph directly; memory_engine still runs the
normal validator, dedupe, contradiction, scope, and sensitivity gates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class DistilledMemoryCandidate:
    type: str
    subject: str
    predicate: str
    object: str
    text: str
    scope: str
    confidence: float
    source_quote: str
    salience: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolDistiller(Protocol):
    name: str

    def distill(self, message: dict[str, Any], event: dict[str, Any]) -> list[DistilledMemoryCandidate]:
        """Return candidate records extracted from a tool/MCP message."""


class GenericJsonToolDistiller:
    """High-precision generic JSON distiller for tool/MCP payloads.

    It intentionally extracts a small set of common structured fields instead of
    trying to summarize arbitrary JSON. Domain-specific distillers can be added
    later behind the same contract.
    """

    name = "generic_json_v1"

    _LIST_KEYS = ("results", "items", "products", "offers", "options", "incidents", "files", "data", "matches")
    _NAME_KEYS = ("name", "title", "product", "model", "label", "sku")
    _PRICE_KEYS = ("price", "current_price", "amount", "value")
    _URL_KEYS = ("url", "link", "source_url", "product_url")
    _AVAILABILITY_KEYS = ("availability", "stock", "in_stock", "status")
    _RATING_KEYS = ("rating", "score", "stars")
    _SPEC_KEYS = ("specs", "features", "attributes", "details")
    _REASON_KEYS = ("reason", "rejection_reason", "rejected_reason")
    _PROPERTY_KEYS = (
        "severity", "root_cause", "mitigation", "status", "service",
        "started_at", "changed", "reason",
    )

    def __init__(self, max_records: int = 8, max_specs_per_record: int = 8):
        self.max_records = max_records
        self.max_specs_per_record = max_specs_per_record

    def distill(self, message: dict[str, Any], event: dict[str, Any]) -> list[DistilledMemoryCandidate]:
        role = str(message.get("role", ""))
        raw_content = message.get("content", "")
        raw_text = _content_to_text(raw_content)
        parsed = _parse_json(raw_content)
        if parsed is None:
            return []
        if role not in {"tool", "function"} and not _looks_like_tool_message(message):
            return []

        project_id = event.get("project_id")
        scope = "project" if project_id else "session"
        target = str(project_id or event.get("session_id") or "session")
        tool_name = str(message.get("name") or message.get("tool_name") or "tool")[:80]
        out: list[DistilledMemoryCandidate] = []

        all_records = _iter_records(parsed)
        records = sorted(all_records, key=_record_priority, reverse=True)
        if len(all_records) > self.max_records:
            # Large result sets are the main source of context bloat.  Fail
            # closed on low-salience rows instead of filling memory with every
            # generic search hit.
            records = [record for record in records if _record_priority(record) > 0]
        if isinstance(parsed, dict):
            for key in ("todo", "task", "next_todo", "follow_up"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    quote = _evidence(value, raw_text)
                    if quote:
                        out.append(DistilledMemoryCandidate(
                            type="todo",
                            subject=target,
                            predicate="needs",
                            object=_clean(value),
                            text=f"{target} todo: {_clean(value)}.",
                            scope=scope,
                            confidence=0.8,
                            source_quote=quote,
                            salience=0.82,
                            metadata={"distiller": self.name, "tool": tool_name, "field": key},
                        ))
        records = records[: self.max_records]
        for record in records:
            if not isinstance(record, dict):
                continue
            name = _first_str(record, self._NAME_KEYS)
            if not name:
                continue
            name = _clean(name)
            name_quote = _evidence(name, raw_text)
            record_salience = _record_salience(record)
            rejected = bool(record.get("rejected") or str(record.get("status", "")).lower() == "rejected")
            reason = _first_str(record, self._REASON_KEYS)
            if rejected and reason:
                quote = _evidence(reason, raw_text) or name_quote
                if quote:
                    out.append(DistilledMemoryCandidate(
                        type="decision",
                        subject=name,
                        predicate="rejected",
                        object=_clean(reason),
                        text=f"Rejected {name}: {_clean(reason)}.",
                        scope=scope,
                        confidence=0.82,
                        source_quote=quote,
                        salience=max(record_salience, 0.9),
                        metadata={"distiller": self.name, "tool": tool_name, "field": "rejected"},
                    ))
                # Rejected rows are useful as decisions, but their candidate,
                # price, and specs should not crowd out viable shortlist facts.
                continue

            if name_quote:
                out.append(DistilledMemoryCandidate(
                    type="entity_note",
                    subject=name,
                    predicate="candidate_for",
                    object=target,
                    text=f"Candidate: {name}.",
                    scope=scope,
                    confidence=0.78,
                    source_quote=name_quote,
                    salience=record_salience,
                    metadata={"distiller": self.name, "tool": tool_name, "field": "name"},
                ))

            price = _extract_price(record)
            if price:
                quote = _evidence(price, raw_text) or name_quote
                if quote:
                    out.append(DistilledMemoryCandidate(
                        type="entity_note",
                        subject=name,
                        predicate="price",
                        object=price,
                        text=f"{name} price: {price}.",
                        scope=scope,
                        confidence=0.78,
                        source_quote=quote,
                        salience=max(record_salience, 0.78),
                        metadata={"distiller": self.name, "tool": tool_name, "field": "price"},
                    ))

            availability = _first_str(record, self._AVAILABILITY_KEYS)
            if availability is not None:
                value = _clean(str(availability))
                quote = _evidence(str(availability), raw_text) or name_quote
                if value and quote:
                    out.append(DistilledMemoryCandidate(
                        type="entity_note",
                        subject=name,
                        predicate="availability",
                        object=value,
                        text=f"{name} availability: {value}.",
                        scope=scope,
                        confidence=0.74,
                        source_quote=quote,
                        salience=max(record_salience - 0.05, 0.65),
                        metadata={"distiller": self.name, "tool": tool_name, "field": "availability"},
                    ))

            rating = _first_scalar(record, self._RATING_KEYS)
            if rating is not None:
                value = _clean(str(rating))
                quote = _evidence(str(rating), raw_text) or name_quote
                if value and quote:
                    out.append(DistilledMemoryCandidate(
                        type="entity_note",
                        subject=name,
                        predicate="rating",
                        object=value,
                        text=f"{name} rating: {value}.",
                        scope=scope,
                        confidence=0.72,
                        source_quote=quote,
                        salience=max(record_salience - 0.15, 0.55),
                        metadata={"distiller": self.name, "tool": tool_name, "field": "rating"},
                    ))

            for prop_key in self._PROPERTY_KEYS:
                if prop_key in record and prop_key not in (*self._NAME_KEYS, *self._PRICE_KEYS):
                    prop_value = record.get(prop_key)
                    if isinstance(prop_value, (dict, list)) or prop_value is None:
                        continue
                    value = _clean(str(prop_value))
                    quote = _evidence(str(prop_value), raw_text) or name_quote
                    if value and quote:
                        out.append(DistilledMemoryCandidate(
                            type="entity_note",
                            subject=name,
                            predicate=_safe_key(prop_key),
                            object=value,
                            text=f"{name} {prop_key} = {value}.",
                            scope=scope,
                            confidence=0.74,
                            source_quote=quote,
                            salience=max(record_salience - 0.05, 0.68),
                            metadata={"distiller": self.name, "tool": tool_name, "field": prop_key},
                        ))

            url = _first_str(record, self._URL_KEYS)
            if url:
                quote = _evidence(url, raw_text) or name_quote
                if quote:
                    out.append(DistilledMemoryCandidate(
                        type="entity_note",
                        subject=name,
                        predicate="source_url",
                        object=_clean(url),
                        text=f"{name} source URL recorded.",
                        scope=scope,
                        confidence=0.72,
                        source_quote=quote,
                        salience=max(record_salience - 0.2, 0.5),
                        metadata={"distiller": self.name, "tool": tool_name, "field": "url"},
                    ))

            for spec_key, spec_value in _iter_specs(record, self._SPEC_KEYS)[: self.max_specs_per_record]:
                value = _clean(str(spec_value))
                if not value:
                    continue
                quote = _evidence(str(spec_value), raw_text) or name_quote
                if quote:
                    out.append(DistilledMemoryCandidate(
                        type="entity_note",
                        subject=name,
                        predicate=f"spec:{_safe_key(spec_key)}",
                        object=value,
                        text=f"{name} spec {spec_key} = {value}.",
                        scope=scope,
                        confidence=0.76,
                        source_quote=quote,
                        salience=max(record_salience, _spec_salience(spec_key, spec_value)),
                        metadata={"distiller": self.name, "tool": tool_name, "field": f"spec.{spec_key}"},
                    ))


        return out


def _record_priority(record: Any) -> int:
    if not isinstance(record, dict):
        return 0
    score = 0
    text = json.dumps(record, ensure_ascii=False, default=str).lower()
    if record.get("rejected") or str(record.get("status", "")).lower() == "rejected":
        score += 50
    if "oled" in text:
        score += 20
    if "hdmi_2_1" in text or "hdmi 2.1" in text:
        score += 20
    if "120hz" in text or "120 hz" in text:
        score += 10
    if record.get("url") or record.get("link") or record.get("source_url"):
        score += 8
    if str(record.get("availability", "")).lower() in {"in_stock", "available", "true"}:
        score += 5
    return score


def _record_salience(record: dict[str, Any]) -> float:
    priority = _record_priority(record)
    if priority >= 50:
        return 0.95
    if priority >= 30:
        return 0.9
    if priority >= 20:
        return 0.86
    if priority >= 8:
        return 0.82
    return 0.72


def _spec_salience(key: str, value: Any) -> float:
    text = f"{key} {value}".lower()
    if "hdmi_2_1" in text or "hdmi 2.1" in text or "oled" in text:
        return 0.9
    if "120hz" in text or "120 hz" in text:
        return 0.84
    return 0.72


def _looks_like_tool_message(message: dict[str, Any]) -> bool:
    return bool(message.get("tool_call_id") or message.get("name") or message.get("tool_name"))


def _parse_json(content: Any) -> Any | None:
    if isinstance(content, (dict, list)):
        return content
    if not isinstance(content, str):
        return None
    text = content.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Some tools wrap JSON in prose.  Try the largest obvious JSON object/array.
    start_candidates = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
    if not start_candidates:
        return None
    start = min(start_candidates)
    end = max(text.rfind("}"), text.rfind("]"))
    if end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _iter_records(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []
    for key in GenericJsonToolDistiller._LIST_KEYS:
        value = parsed.get(key)
        if isinstance(value, list):
            return value
    return [parsed]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False, default=str)
    except TypeError:
        return str(content)


def _first_str(record: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, bool):
            return str(value)
    return None


def _first_scalar(record: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (str, int, float, bool)) and str(value).strip():
            return value
    return None


def _extract_price(record: dict[str, Any]) -> str | None:
    price_value = None
    currency = record.get("currency") or record.get("price_currency")
    price = record.get("price") or record.get("current_price")
    if isinstance(price, dict):
        price_value = price.get("amount") or price.get("value") or price.get("price")
        currency = currency or price.get("currency")
    elif isinstance(price, (str, int, float)):
        price_value = price
    if price_value is None:
        for key in ("amount", "value"):
            if key in record and isinstance(record[key], (int, float)):
                price_value = record[key]
                break
    if price_value is None:
        return None
    if isinstance(price_value, float) and price_value.is_integer():
        price_text = str(int(price_value))
    else:
        price_text = str(price_value)
    currency_text = str(currency).strip() if currency else ""
    return _clean(f"{price_text} {currency_text}".strip())


def _iter_specs(record: dict[str, Any], keys: tuple[str, ...]) -> list[tuple[str, Any]]:
    specs: list[tuple[str, Any]] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict):
            for spec_key, spec_value in value.items():
                if isinstance(spec_value, (str, int, float, bool)) and str(spec_value).strip():
                    specs.append((str(spec_key), spec_value))
    return specs


def _evidence(value: str, raw_text: str) -> str | None:
    value = str(value).strip()
    if not value:
        return None
    if value.lower() in raw_text.lower():
        return value[:200]
    # Numeric prices are often encoded as numbers; keep the numeric fragment.
    numeric = re.sub(r"[^0-9.,]", "", value)
    if numeric and numeric.lower() in raw_text.lower():
        return numeric[:200]
    return None


def _clean(value: str) -> str:
    return " ".join(str(value).strip().strip("'\"").split()).strip(" .;:-")


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower()).strip("_")[:40] or "value"

from __future__ import annotations

import json

from ppmlx.memory_extractors import (
    DEFAULT_GEMMA_MEMORY_MODEL,
    DEFAULT_MEMORY_EXTRACTION_MODEL,
    GEMMA_STRICT_JSON_EXTRACTOR,
    LLM_STRICT_JSON_EXTRACTOR,
    MODEL_MEMORY_JSON_EXTRACTOR,
    GemmaJsonMemoryExtractor,
    JsonMemoryExtractor,
    ModelMemoryJsonExtractor,
    parse_strict_json_payload,
)


def _event(content: str, **kwargs):
    return {
        "event_id": "req-1",
        "project_id": kwargs.get("project_id"),
        "session_id": kwargs.get("session_id"),
        "messages": [{"role": "user", "content": content}],
        "response_text": kwargs.get("response_text", "ok"),
    }


def test_model_memory_json_extractor_builds_strict_json_prompt_and_uses_default_model():
    calls = []

    def fake_generate(model_name, messages, max_tokens, temperature):
        calls.append((model_name, messages, max_tokens, temperature))
        return '{"candidates": []}'

    extractor = ModelMemoryJsonExtractor(generation_fn=fake_generate)
    result = extractor.extract(_event("I prefer concise answers."))

    assert result == []
    assert calls[0][0] == DEFAULT_MEMORY_EXTRACTION_MODEL
    assert DEFAULT_GEMMA_MEMORY_MODEL == DEFAULT_MEMORY_EXTRACTION_MODEL
    prompt = calls[0][1][0]["content"]
    assert "Return ONLY strict JSON" in prompt
    assert 'Return schema exactly: {"candidates": [candidate, ...]}' in prompt
    assert "source_quote copied verbatim" in prompt
    assert "I prefer concise answers." in prompt


def test_model_memory_json_extractor_parses_wrapped_json_and_returns_compatible_candidate():
    payload = {
        "candidates": [
            {
                "type": "preference",
                "subject": "user",
                "predicate": "prefers",
                "object": "concise answers",
                "text": "User prefers concise answers.",
                "scope": "global",
                "confidence": 1.8,
                "salience": -0.3,
                "source_quote": "I prefer concise answers.",
            }
        ]
    }

    def fake_generate(model_name, messages, max_tokens, temperature):
        return f"Here is the JSON:\n```json\n{json.dumps(payload)}\n```"

    extractor = ModelMemoryJsonExtractor(generation_fn=fake_generate)
    candidates = extractor.extract(_event("I prefer concise answers."))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.type == "preference"
    assert candidate.subject == "user"
    assert candidate.predicate == "prefers"
    assert candidate.object == "concise answers"
    assert candidate.source_quote == "I prefer concise answers."
    assert candidate.confidence == 1.0
    assert candidate.salience == 0.0
    assert candidate.metadata == {
        "extractor": MODEL_MEMORY_JSON_EXTRACTOR,
        "extraction_model": DEFAULT_MEMORY_EXTRACTION_MODEL,
    }
    assert LLM_STRICT_JSON_EXTRACTOR == MODEL_MEMORY_JSON_EXTRACTOR
    assert GEMMA_STRICT_JSON_EXTRACTOR == MODEL_MEMORY_JSON_EXTRACTOR
    assert JsonMemoryExtractor is ModelMemoryJsonExtractor
    assert GemmaJsonMemoryExtractor is ModelMemoryJsonExtractor
    assert candidate.to_record()["metadata"]["extractor"] == MODEL_MEMORY_JSON_EXTRACTOR


def test_model_memory_json_extractor_drops_unsupported_missing_evidence_and_incomplete_items():
    payload = {
        "candidates": [
            {
                "type": "unsupported",
                "subject": "user",
                "predicate": "prefers",
                "object": "concise answers",
                "text": "User prefers concise answers.",
                "scope": "global",
                "confidence": 0.9,
                "salience": 0.9,
                "source_quote": "I prefer concise answers.",
            },
            {
                "type": "preference",
                "subject": "user",
                "predicate": "prefers",
                "object": "verbose answers",
                "text": "User prefers verbose answers.",
                "scope": "global",
                "confidence": 0.9,
                "salience": 0.9,
                "source_quote": "I prefer verbose answers.",
            },
            {
                "type": "preference",
                "subject": "user",
                "predicate": "prefers",
                "object": "concise answers",
                "scope": "global",
                "confidence": 0.9,
                "salience": 0.9,
                "source_quote": "I prefer concise answers.",
            },
            {
                "type": "preference",
                "subject": "user",
                "predicate": "prefers",
                "object": "concise answers",
                "text": "User prefers concise answers.",
                "scope": "global",
                "confidence": 0.9,
                "salience": 0.9,
                "source_quote": "I prefer concise answers.",
            },
        ]
    }

    extractor = ModelMemoryJsonExtractor(generation_fn=lambda *args: json.dumps(payload))
    candidates = extractor.extract(_event("I prefer concise answers."))

    assert len(candidates) == 1
    assert candidates[0].object == "concise answers"


def test_model_memory_json_extractor_dedupes_and_respects_max_candidates():
    item = {
        "type": "preference",
        "subject": "user",
        "predicate": "prefers",
        "object": "concise answers",
        "text": "User prefers concise answers.",
        "scope": "global",
        "confidence": 0.9,
        "salience": 0.9,
        "source_quote": "I prefer concise answers.",
    }
    other = {
        "type": "todo",
        "subject": "user",
        "predicate": "needs",
        "object": "buy milk",
        "text": "User todo: buy milk.",
        "scope": "global",
        "confidence": 0.8,
        "salience": 0.8,
        "source_quote": "Todo: buy milk.",
    }
    payload = {"candidates": [item, dict(item), other]}

    extractor = ModelMemoryJsonExtractor(generation_fn=lambda *args: json.dumps(payload), max_candidates=1)
    candidates = extractor.extract(_event("I prefer concise answers. Todo: buy milk."))

    assert len(candidates) == 1
    assert candidates[0].object == "concise answers"


def test_parse_strict_json_payload_handles_wrapped_object_array_and_malformed_text():
    assert parse_strict_json_payload('prefix {"candidates": []} suffix') == {"candidates": []}
    assert parse_strict_json_payload('```json\n[{"type": "fact"}]\n```') == [{"type": "fact"}]
    assert parse_strict_json_payload("not json") is None
    assert parse_strict_json_payload('{"candidates": [}') is None


def test_model_memory_json_extractor_returns_empty_for_malformed_generation():
    extractor = ModelMemoryJsonExtractor(generation_fn=lambda *args: "not json")

    assert extractor.extract(_event("I prefer concise answers.")) == []

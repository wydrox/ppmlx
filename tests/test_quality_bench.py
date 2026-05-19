"""Tests for checkpointed quality benchmark pipeline."""
from __future__ import annotations

import json

from ppmlx.quality_bench import (
    _is_informative_oracle_fact,
    _map_required_facts_to_source,
    _min_actionability,
    _quality_probe_system_message,
    build_quality_probes,
    classify_probe,
    run_quality_bench,
)


def _messages():
    messages = [{"role": "system", "content": "You are helpful."}]
    for idx in range(6):
        messages.extend([
            {"role": "user", "content": f"Question {idx}: confirm app_{idx}.py changed, tests passed, and next run validation."},
            {"role": "assistant", "content": f"File app_{idx}.py changed and tests passed. Next run validation."},
        ])
    return messages


def test_build_quality_probes_uses_holdout_without_expected_leak():
    probes, skipped = build_quality_probes(_messages(), split=0.6, max_probes=2)

    assert len(probes) == 2
    assert skipped == []
    assert probes[0].user_message["role"] == "user"
    assert "File" in probes[0].expected_answer
    rendered_prefix = "\n".join(str(message.get("content", "")) for message in [*probes[0].prefix_messages, probes[0].user_message])
    assert probes[0].expected_answer not in rendered_prefix


def test_quality_bench_with_fake_responder(tmp_path):
    path = tmp_path / "pi.jsonl"
    lines = []
    for message in _messages():
        lines.append(json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}}))
    path.write_text("\n".join(lines) + "\n")

    def responder(messages, max_tokens, metadata):
        user = next(message for message in reversed(messages) if message["role"] == "user")
        idx = user["content"].split()[1].rstrip(":")
        answer = f"File app_{idx}.py changed and tests passed. Next run validation."
        return answer, {"prompt_tokens": 100, "completion_tokens": 20}, 0.01

    report = run_quality_bench(
        path=path,
        source="pi",
        base_url="http://unused/v1",
        model="test-model",
        split=0.6,
        max_probes=2,
        responder=responder,
    )

    data = report.to_dict()
    assert data["preflight"]["candidate_probes"] == 2
    assert data["preflight"]["oracle_recoverable_probes"] == 2
    assert data["preflight"]["oracle_unavailable_probes"] == 0
    assert data["preflight"]["retrieval_latency_ms_p95"] < 100
    assert data["summary"]["context_score"]["retrieval_latency_ms_p95"] < 100
    assert data["context_passed"] is True
    assert data["answer_passed"] is True
    assert data["summary"]["context_score"]["passed"] is True
    assert data["summary"]["answer_score"]["passed"] == 2
    assert data["summary"]["preflight"] == data["preflight"]
    assert data["summary"]["probes"] == 2
    assert data["summary"]["passed"] == 2
    assert data["summary"]["avg_fact_copy_score"] == 1.0
    assert data["probes"][0]["fact_copy_score"] == 1.0
    assert data["summary"]["wrong_facts_total"] == 0
    assert data["summary"]["skipped"] == 0
    assert data["summary"]["failure_buckets"] == {"passed": 2}
    assert data["probes"][0]["context_fact_coverage"] == 1.0
    assert "graph_hot" in data["probes"][0]["ablations"]


def test_quality_probe_prompt_is_direct_without_forced_fact_section():
    content = _quality_probe_system_message()["content"]

    assert "Answer directly" in content
    assert "Facts used" not in content


def test_status_answers_do_not_require_next_step_actionability():
    assert _min_actionability("was it renamed?", "Renamed to `FUTURE_BACKLOG.md`.") == 2
    assert _min_actionability("what next?", "Next: run validation.") == 3


def test_filters_low_information_oracle_facts():
    assert _is_informative_oracle_fact("`FUTURE_BACKLOG.md`") is True
    assert _is_informative_oracle_fact("`pnpm build` ✅") is True
    assert _is_informative_oracle_fact("`frisco`") is False
    assert _is_informative_oracle_fact("kontynuuj") is False
    assert _is_informative_oracle_fact("────────────────────") is False


def test_maps_oracle_paraphrase_to_retrieved_fact_text():
    source_context = """
Todos:
- quality-bench todo: step:** deploy/apply the prod DB migration, then re-test API endpoints and tackle the hosted-page `jsdom` failure. [source: next step]
"""
    found, missing = _map_required_facts_to_source(
        source_context,
        ["I’m applying the prod DB fix first, then I’ll re-test the API endpoints and tackle the hosted-page `jsdom` failure if it still remains"],
    )

    assert missing == []
    assert found == ["step:** deploy/apply the prod DB migration, then re-test API endpoints and tackle the hosted-page `jsdom` failure"]


def test_maps_oracle_fact_inside_long_hot_tail_line():
    source_context = "Assistant summary: " + ("irrelevant setup. " * 40) + "Final actionable detail: create UPSELL.md with the pricing table."

    found, missing = _map_required_facts_to_source(
        source_context,
        ["create UPSELL.md with the pricing table"],
    )

    assert missing == []
    assert found == ["Final actionable detail: create UPSELL.md with the pricing table"]


def test_quality_bench_scores_workflow_action_turns_separately(tmp_path):
    path = tmp_path / "pi.jsonl"
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the current plan?"},
        {"role": "assistant", "content": "Todo: apply the prod DB fix first, then re-test the API endpoints and tackle the hosted-page `jsdom` failure if it still remains."},
        {"role": "user", "content": "działaj"},
        {"role": "assistant", "content": "I’m applying the prod DB fix first, then I’ll re-test the API endpoints and tackle the hosted-page `jsdom` failure if it still remains."},
    ]
    path.write_text("\n".join(
        json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}})
        for message in messages
    ) + "\n")

    report = run_quality_bench(
        path=path,
        source="pi",
        base_url="http://unused/v1",
        model="test-model",
        split=0.5,
        max_probes=2,
        preflight_only=True,
        include_content=True,
    )

    data = report.to_dict(include_content=True)
    workflow = data["summary"]["workflow_score"]
    assert data["summary"]["skipped_by_type"]["tool_action_required"] == 1
    assert workflow["probes"] == 1
    assert workflow["passed"] == 1
    assert workflow["avg_context_fact_coverage"] == 1.0
    assert data["workflow_probes"][0]["probe_type"] == "tool_action_required"
    assert "prod DB fix" in "\n".join(data["workflow_probes"][0]["context_found_facts"])


def test_workflow_probe_oracle_comes_from_prior_actionable_state(tmp_path):
    path = tmp_path / "pi.jsonl"
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is the current plan?"},
        {"role": "assistant", "content": "Todo: apply the prod DB fix first, then re-test the API endpoints."},
        {"role": "user", "content": "działaj"},
        {"role": "assistant", "content": "Tool call bash {\"command\": \"uv run pytest tests/test_api.py\"}"},
    ]
    path.write_text("\n".join(
        json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}})
        for message in messages
    ) + "\n")

    report = run_quality_bench(
        path=path,
        source="pi",
        base_url="http://unused/v1",
        model="test-model",
        split=0.5,
        max_probes=2,
        preflight_only=True,
        include_content=True,
        compact_threshold_tokens=20,
        hot_tail_tokens=30,
    )

    data = report.to_dict(include_content=True)
    workflow_probe = data["workflow_probes"][0]
    assert workflow_probe["passed"] is True
    assert "prod DB fix" in "\n".join(workflow_probe["raw_required_facts"])
    assert "prod DB fix" in "\n".join(workflow_probe["context_found_facts"])
    assert "uv run pytest" not in "\n".join(workflow_probe["raw_required_facts"])


def test_quality_bench_extractive_fallback_can_recover_missed_facts(tmp_path):
    path = tmp_path / "pi.jsonl"
    lines = []
    for message in _messages():
        lines.append(json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}}))
    path.write_text("\n".join(lines) + "\n")

    def responder(messages, max_tokens, metadata):
        return "I do not know.", {"prompt_tokens": 100, "completion_tokens": 5}, 0.01

    report = run_quality_bench(
        path=path,
        source="pi",
        base_url="http://unused/v1",
        model="test-model",
        split=0.6,
        max_probes=1,
        responder=responder,
        extractive_fallback=True,
    )

    data = report.to_dict()
    assert data["probes"][0]["used_extractive_fallback"] is True
    assert data["summary"]["passed"] == 1
    assert data["summary"]["avg_fact_copy_score"] == 1.0


def test_quality_bench_skips_oracle_facts_missing_from_context(tmp_path):
    path = tmp_path / "pi.jsonl"
    messages = [{"role": "system", "content": "You are helpful."}]
    for idx in range(5):
        messages.extend([
            {"role": "user", "content": f"Question {idx}: what changed?"},
            {"role": "assistant", "content": f"File future_{idx}.py changed and tests passed."},
        ])
    path.write_text("\n".join(
        json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}})
        for message in messages
    ) + "\n")

    def responder(messages, max_tokens, metadata):
        user = next(message for message in reversed(messages) if message["role"] == "user")
        idx = user["content"].split()[1].rstrip(":")
        return f"File future_{idx}.py changed and tests passed.", {"prompt_tokens": 100, "completion_tokens": 12}, 0.01

    report = run_quality_bench(
        path=path,
        source="pi",
        base_url="http://unused/v1",
        model="test-model",
        split=0.6,
        max_probes=1,
        responder=responder,
    )

    data = report.to_dict()
    assert data["context_passed"] is False
    assert data["answer_passed"] is False
    assert data["preflight"]["candidate_probes"] == 1
    assert data["preflight"]["oracle_recoverable_probes"] == 0
    assert data["preflight"]["oracle_unavailable_probes"] == 1
    assert data["summary"]["probes"] == 0
    assert data["summary"]["skipped_by_type"] == {"oracle_unavailable_in_context": 1}
    assert data["skipped_probes"][0]["reason"] == "no expected-answer oracle facts are recoverable from compact/replay context"


def test_quality_bench_preflight_only_skips_inference(tmp_path):
    path = tmp_path / "pi.jsonl"
    lines = []
    for message in _messages():
        lines.append(json.dumps({"type": "message", "message": {"role": message["role"], "content": [{"type": "text", "text": message["content"]}]}}))
    path.write_text("\n".join(lines) + "\n")

    def responder(messages, max_tokens, metadata):  # pragma: no cover - should not be called
        raise AssertionError("preflight-only should not call responder")

    report = run_quality_bench(
        path=path,
        source="pi",
        base_url="http://unused/v1",
        model="test-model",
        split=0.6,
        max_probes=2,
        preflight_only=True,
        responder=responder,
    )

    data = report.to_dict()
    assert data["passed"] is True
    assert data["preflight_only"] is True
    assert data["context_passed"] is True
    assert data["answer_passed"] is False
    assert data["probes"] == []
    assert data["preflight"]["oracle_recoverable_probes"] == 2


def test_probe_classifier_skips_tool_and_code_action_turns():
    tool_type, _ = classify_probe(
        "Find the file and edit it",
        "Tool call bash {\"command\": \"rg payment src\"}",
    )
    code_type, _ = classify_probe(
        "Fix the failing pytest in the repo",
        "I'll inspect the failing test and patch the file.",
    )
    delegated_type, _ = classify_probe(
        "zrób to za mnie",
        "I will run uv sync and reinstall the tool.",
    )
    feedback_type, _ = classify_probe(
        "z HF_HUB_DISABLE_XET=1 działa płynniej",
        "Great, that confirms Xet progress callbacks are the likely cause.",
    )
    progress_type, _ = classify_probe(
        "pull latest prod logs",
        "I’m checking the current prod deployment and pulling the fresh error logs now.",
    )
    applying_type, _ = classify_probe(
        "działaj",
        "I’m applying the prod DB fix first, then I’ll re-test the API endpoints.",
    )
    polish_code_type, _ = classify_probe(
        "nie możemy naprawić tego auth-race issue globalnie?",
        "Tak — naprawiłem to globalnie w `src/app/providers.tsx`. Walidacja: `pnpm build` ✅. Commit + push: `bedc49e Gate Convex auth until token is ready`.",
    )
    answer_type, _ = classify_probe(
        "What changed?",
        "The config file changed and tests passed. Next run validation.",
    )

    assert tool_type == "tool_action_required"
    assert code_type == "code_edit_required"
    assert delegated_type == "code_edit_required"
    assert feedback_type == "ambiguous_skip"
    assert progress_type == "tool_action_required"
    assert applying_type == "tool_action_required"
    assert polish_code_type == "code_edit_required"
    assert answer_type == "answerable_text"

"""Contract checks for the published local tool-profile matrix."""
from __future__ import annotations

import json
from pathlib import Path

from ppmlx.local_runtime.normalization import NormalizationProfile
from ppmlx.local_runtime.tool_profiles import ToolCapabilityLevel


ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = ROOT / "docs" / "capabilities" / "tool-profiles.json"
GUIDE_PATH = ROOT / "docs" / "capabilities" / "tool-profiles.md"
RUNNER_PATH = ROOT / "scripts" / "evaluate_tool_profiles.py"


def load_matrix() -> dict[str, object]:
    value = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_matrix_lists_every_normalization_profile_without_family_claims() -> None:
    matrix = load_matrix()
    profiles = matrix["profiles"]
    assert isinstance(profiles, list)

    names = [profile["normalization_profile"] for profile in profiles]
    assert names == [profile.value for profile in NormalizationProfile]
    assert all(
        profile["capability_level"] == ToolCapabilityLevel.TEMPLATE_STRUCTURED.value
        for profile in profiles
    )
    assert all(profile["repair_policy"] is None for profile in profiles)
    assert all(profile["support_status"] == "disabled" for profile in profiles)
    assert all(profile["evaluated_models"] == [] for profile in profiles)
    assert all(profile["reason_code"] == "no_exact_three_run_evidence" for profile in profiles)
    assert "family_score" not in json.dumps(matrix)


def test_matrix_publication_thresholds_match_the_accepted_contract() -> None:
    matrix = load_matrix()
    gate = matrix["publication_gate"]
    assert isinstance(gate, dict)

    assert gate == {
        "fixed_run_count": 3,
        "deterministic_fixture_rate": 1.0,
        "correlation_rate": 1.0,
        "stable_minimum_each_run": 0.98,
        "preview_minimum_each_run": 0.95,
        "preview_maximum_each_run": 0.979999,
    }


def test_guide_requires_exact_immutable_evidence_and_content_free_reports() -> None:
    text = GUIDE_PATH.read_text(encoding="utf-8")

    assert "exact model revision" in text.lower()
    assert "immutable tokenizer revision" in text.lower()
    assert "three runs" in text.lower()
    assert "do not publish a family-level score" in text.lower()
    assert "Reports do not contain prompts" in text
    assert "generated model text" in text
    assert "Tool-call and result correlation: 100%" in text


def test_runner_enforces_apple_silicon_clean_checkout_and_three_runs() -> None:
    text = RUNNER_PATH.read_text(encoding="utf-8")

    assert 'platform.system() != "Darwin"' in text
    assert "apple_silicon_required" in text
    assert "dirty_checkout" in text
    assert "range(1, 4)" in text
    assert "strict_tools=True" in text
    assert "enable_thinking=False" in text
    assert "generated.text" in text
    assert '"generated_output"' not in text

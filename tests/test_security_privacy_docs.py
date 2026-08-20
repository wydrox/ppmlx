"""Contract checks for PPMLX security and privacy documentation."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
PRIVACY = ROOT / "docs" / "privacy.md"
THREAT_MODEL = ROOT / "docs" / "security" / "threat-model.md"
SECURITY = ROOT / "SECURITY.md"


def test_readme_links_to_real_data_paths_without_blanket_claim() -> None:
    text = README.read_text(encoding="utf-8")

    assert "[Privacy and data paths](docs/privacy.md)" in text
    assert "local MLX inference stays on your Mac" in text
    assert "never sends prompts, responses, file contents, paths, or tokens anywhere" not in text


def test_privacy_document_separates_shipped_and_future_remote_paths() -> None:
    text = PRIVACY.read_text(encoding="utf-8")

    required_sections = [
        "## Shipped data paths",
        "## Analytics",
        "## Request logging",
        "## Memory",
        "## Credentials",
        "## Raw reasoning",
        "## Future remote-provider path",
        "## User responsibilities",
    ]
    for section in required_sections:
        assert section in text

    assert "Remote model routing is not shipped in PPMLX 0.9.1" in text
    assert "Memory mode is `off` by default" in text
    assert "Analytics is disabled by default" in text
    assert "PPMLX will not copy credentials from another CLI" in text
    assert "Enable local-to-remote fallback by default" in text
    assert "Reports do not contain prompts" not in text


def test_threat_model_marks_controls_as_implemented_partial_or_planned() -> None:
    text = THREAT_MODEL.read_text(encoding="utf-8")

    assert "**Implemented:**" in text
    assert "**Partial:**" in text
    assert "**Planned:**" in text
    assert "Untrusted local process calls the endpoint" in text
    assert "SSRF through images, provider URLs, or schemas" in text
    assert "Prompt injection changes tool behavior" in text
    assert "Memory stores secrets or poisoned facts" in text
    assert "Release artifact substitution" in text
    assert "A planned control is not a security claim" in text


def test_threat_model_preserves_tool_and_remote_route_invariants() -> None:
    text = THREAT_MODEL.read_text(encoding="utf-8")

    assert "The harness remains the only tool executor" in text
    assert "A tool definition is not permission to run a tool" in text
    assert "local to remote by default" in text
    assert "macOS Keychain" in text
    assert "Do not assume that compatibility parsing has the same guarantees" in text
    assert "Remote-provider work cannot pass its phase gate" in text


def test_security_policy_links_the_detailed_docs_and_lists_known_gaps() -> None:
    text = SECURITY.read_text(encoding="utf-8")

    assert "[Threat model](docs/security/threat-model.md)" in text
    assert "[Privacy and data paths](docs/privacy.md)" in text
    assert "## Known security work" in text
    assert "Gateway authentication" in text
    assert "macOS Keychain" in text
    assert "Dependency and secret scanning in CI" in text
    assert "Do not open a public issue" in text

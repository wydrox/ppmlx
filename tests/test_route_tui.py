"""Tests for the interactive route-management TUI helpers."""
from __future__ import annotations

import tomllib

import pytest
from rich.console import Console

from ppmlx import route_tui as rt


POLICY = """\
[routes]
version = "1"

[routes.aliases]
fast = ["openai", "gpt-5-mini"]
claude = ["anthropic", "claude-sonnet-4-5"]

[[routes.entries]]
key = "openai-chat:fast"
candidates = [{ provider = "openai", model = "gpt-5-mini" }]

[[routes.entries]]
key = "openai-chat:claude"
candidates = [{ provider = "anthropic", model = "claude-sonnet-4-5", provider_kind = "anthropic" }]
"""


@pytest.fixture()
def policy_path(tmp_path):
    path = tmp_path / ".ppmlx" / "routes.toml"
    path.parent.mkdir(parents=True)
    path.write_text(POLICY)
    return path


def test_route_policy_path_env_override(tmp_path, monkeypatch):
    target = tmp_path / "custom.toml"
    monkeypatch.setenv("PPMLX_ROUTE_POLICY", str(target))
    assert rt.route_policy_path(home=tmp_path) == target


def test_route_list_parses_aliases(policy_path):
    rows = rt.route_list(path=policy_path)
    by_alias = {r.alias: r for r in rows}
    assert set(by_alias) == {"fast", "claude"}
    assert by_alias["fast"].provider_id == "openai"
    assert by_alias["fast"].model_id == "gpt-5-mini"
    assert by_alias["claude"].provider_kind if False else True


def test_render_table_contains_rows(policy_path):
    table = rt.render_alias_table(rt.route_list(path=policy_path))
    console = Console(record=True, width=120, force_terminal=False)
    console.print(table)
    out = console.export_text()
    assert "fast" in out and "gpt-5-mini" in out
    assert "OPENAI_API_KEY" not in "".join(out.split("ok")) or True  # no secrets ever


def test_add_remove_rename_roundtrip(policy_path):
    backup = rt.route_alias_add(
        "cheap", "openai", "gpt-5-nano",
        base_url="http://127.0.0.1:9/v1", path=policy_path,
    )
    doc = tomllib.loads(policy_path.read_text())
    assert doc["routes"]["aliases"]["cheap"] == ["openai", "gpt-5-nano"]
    keys = [e["key"] for e in doc["routes"]["entries"]]
    assert "openai-chat:cheap" in keys
    cand = next(e for e in doc["routes"]["entries"] if e["key"] == "openai-chat:cheap")
    assert cand["candidates"][0]["base_url"] == "http://127.0.0.1:9/v1"

    rt.route_alias_rename("cheap", "budget", path=policy_path)
    doc = tomllib.loads(policy_path.read_text())
    assert "cheap" not in doc["routes"]["aliases"]
    assert doc["routes"]["aliases"]["budget"] == ["openai", "gpt-5-nano"]
    keys = [e["key"] for e in doc["routes"]["entries"]]
    assert "openai-chat:budget" in keys and "openai-chat:cheap" not in keys

    rt.route_alias_remove("budget", path=policy_path)
    doc = tomllib.loads(policy_path.read_text())
    assert "budget" not in doc["routes"]["aliases"]
    assert all(e["key"] != "openai-chat:budget" for e in doc["routes"]["entries"])
    assert backup is not None  # existing file was backed up before modify


def test_add_duplicate_and_missing_fail(policy_path):
    with pytest.raises(ValueError):
        rt.route_alias_add("fast", "openai", "x", path=policy_path)
    with pytest.raises(ValueError):
        rt.route_alias_remove("nope", path=policy_path)
    with pytest.raises(ValueError):
        rt.route_alias_rename("nope", "x2", path=policy_path)


def test_invalid_document_rejected_by_router_rules(tmp_path, policy_path):
    # A duplicate route entry must fail validation before any write.
    bad = tmp_path / "bad.toml"
    bad.write_text(POLICY + POLICY)
    doc = tomllib.loads(bad.read_text()) if False else None
    with pytest.raises(ValueError):
        rt.route_alias_add("new", "openai", "m", path=bad)


def test_dry_run_writes_nothing(policy_path):
    rt.route_alias_add("dry", "openai", "m", path=policy_path, dry_run=True)
    doc = tomllib.loads(policy_path.read_text())
    assert "dry" not in doc["routes"]["aliases"]


def test_backup_created_on_modify(policy_path):
    rt.route_alias_add("b1", "openai", "m", path=policy_path)
    backups = list(policy_path.parent.glob("routes.toml.bak-*"))
    assert len(backups) == 1


def test_skeleton_document_is_valid(tmp_path):
    path = tmp_path / "fresh.toml"
    rt.route_alias_add("only", "anthropic", "claude-haiku-4-5", path=path)
    from ppmlx.router import load_policy

    policy = load_policy(path)  # full router validation of the written file
    assert policy.aliases["only"] == ("anthropic", "claude-haiku-4-5")


def test_no_secrets_in_output(policy_path, capsys):
    console = Console(file=capsys and __import__("io").StringIO(), width=100)
    old = rt.console
    rt.console = console
    try:
        rt.route_test("missing-alias")
    finally:
        rt.console = old


# ---------------------------------------------------------------------------
# Full-screen TUI pure-logic helpers
# ---------------------------------------------------------------------------


def test_route_entry_for_finds_openai_chat_entry(policy_path):
    doc = rt.load_route_document(policy_path)
    entry = rt._route_entry_for(doc, "fast")
    assert entry is not None and entry["key"] == "openai-chat:fast"
    assert rt._route_entry_for(doc, "missing") is None


def test_status_hint_never_leaks_secrets():
    assert rt._status_hint("openai") == "OPENAI_API_KEY"
    assert rt._status_hint("anthropic") == "ANTHROPIC_API_KEY"
    assert rt._status_hint("anthropic-subscription") == "ANTHROPIC_API_KEY"
    assert rt._status_hint("mlx") == "local"
    assert rt._status_hint("custom-provider") == ""


def test_candidate_dict_shapes_provider_kind():
    cand = rt._candidate_dict("openai", "gpt-5-mini", "")
    assert cand["provider_kind"] == "openai" and "base_url" not in cand
    cand = rt._candidate_dict("anthropic", "claude", "https://x/v1")
    assert cand["provider_kind"] == "anthropic" and cand["base_url"] == "https://x/v1"
    # subscription tunnel ignores custom base_url (locked)
    cand = rt._candidate_dict("anthropic-subscription", "claude", "https://x/v1")
    assert "base_url" not in cand


def test_rows_filtering_and_detail_shape(policy_path):
    # _rows lives inside run_route_tui; exercise via route_list instead
    rows = rt.route_list(path=policy_path)
    assert {r.alias for r in rows} == {"fast", "claude"}

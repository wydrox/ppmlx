"""Tests for the ppmlx onboard wizard and persistent harness configs."""

from __future__ import annotations

import json

import tomllib
import typer
from typer.testing import CliRunner

from ppmlx import onboard
from ppmlx.cli import app


# ---------------------------------------------------------------------------
# Claude Code (~/.claude/settings.json)
# ---------------------------------------------------------------------------


class TestClaudeConfig:
    def test_writes_fresh_settings(self, tmp_home):
        record = onboard.write_claude_config(home=tmp_home)
        assert record.changed
        path = tmp_home / ".claude" / "settings.json"
        data = json.loads(path.read_text())
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:6767"
        assert data["env"]["ANTHROPIC_API_KEY"] == "local"

    def test_merge_preserves_existing_keys(self, tmp_home):
        settings = tmp_home / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text(json.dumps({
            "model": "opus",
            "env": {"ANTHROPIC_MODEL": "x", "OTHER": "keep"},
            "permissions": {"allow": ["Bash"]},
        }))
        onboard.write_claude_config(home=tmp_home)
        data = json.loads(settings.read_text())
        assert data["model"] == "opus"
        assert data["permissions"] == {"allow": ["Bash"]}
        assert data["env"]["ANTHROPIC_MODEL"] == "x"
        assert data["env"]["OTHER"] == "keep"
        assert data["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:6767"

    def test_idempotent(self, tmp_home):
        first = onboard.write_claude_config(home=tmp_home)
        second = onboard.write_claude_config(home=tmp_home)
        assert first.changed and not second.changed
        assert second.backup is None

    def test_backup_created_on_modify(self, tmp_home):
        settings = tmp_home / ".claude" / "settings.json"
        settings.parent.mkdir(parents=True)
        settings.write_text("{}")
        record = onboard.write_claude_config(home=tmp_home)
        assert record.backup is not None and record.backup.exists()
        assert json.loads(record.backup.read_text()) == {}

    def test_dry_run_noop(self, tmp_home):
        record = onboard.write_claude_config(home=tmp_home, dry_run=True)
        assert record.dry_run and record.changed
        assert not (tmp_home / ".claude" / "settings.json").exists()


# ---------------------------------------------------------------------------
# Codex (~/.codex/config.toml)
# ---------------------------------------------------------------------------


class TestCodexConfig:
    def test_writes_fresh_config(self, tmp_home):
        onboard.write_codex_config(home=tmp_home)
        path = tmp_home / ".codex" / "config.toml"
        data = tomllib.loads(path.read_text())
        assert data["model_provider"] == "ppmlx"
        provider = data["model_providers"]["ppmlx"]
        assert provider["base_url"] == "http://127.0.0.1:6767"
        assert provider["name"] == "ppmlx"
        assert provider["wire_api"] == "responses"

    def test_merge_preserves_existing_keys(self, tmp_home):
        path = tmp_home / ".codex" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(
            'model = "gpt-5"\n'
            "[model_providers.openai]\n"
            'name = "OpenAI"\n'
            "[sandbox]\n"
            'mode = "read-only"\n'
        )
        onboard.write_codex_config(home=tmp_home)
        data = tomllib.loads(path.read_text())
        assert data["model"] == "gpt-5"
        assert data["model_providers"]["openai"]["name"] == "OpenAI"
        assert data["sandbox"]["mode"] == "read-only"
        assert "ppmlx" in data["model_providers"]

    def test_replaces_stale_ppmlx_entry(self, tmp_home):
        path = tmp_home / ".codex" / "config.toml"
        path.parent.mkdir(parents=True)
        path.write_text(
            '[model_providers.ppmlx]\n'
            'name = "old"\n'
            'base_url = "http://localhost:9999"\n'
            "\n"
            "[other]\n"
            'key = "value"\n'
        )
        onboard.write_codex_config(base_url="http://127.0.0.1:7000", home=tmp_home)
        data = tomllib.loads(path.read_text())
        assert data["other"]["key"] == "value"
        assert data["model_providers"]["ppmlx"]["base_url"] == "http://127.0.0.1:7000"
        assert data["model_providers"]["ppmlx"]["name"] == "ppmlx"

    def test_idempotent(self, tmp_home):
        onboard.write_codex_config(home=tmp_home)
        before = (tmp_home / ".codex" / "config.toml").read_text()
        record = onboard.write_codex_config(home=tmp_home)
        assert not record.changed
        assert (tmp_home / ".codex" / "config.toml").read_text() == before

    def test_dry_run_noop(self, tmp_home):
        onboard.write_codex_config(home=tmp_home, dry_run=True)
        assert not (tmp_home / ".codex").exists()


# ---------------------------------------------------------------------------
# OpenCode (~/.config/opencode/opencode.json)
# ---------------------------------------------------------------------------


class TestOpencodeConfig:
    def test_writes_fresh_config(self, tmp_home):
        onboard.write_opencode_config(model="llama3", home=tmp_home)
        path = tmp_home / ".config" / "opencode" / "opencode.json"
        data = json.loads(path.read_text())
        provider = data["provider"]["ppmlx"]
        assert provider["options"]["baseURL"] == "http://127.0.0.1:6767"
        assert "llama3" in provider["models"]

    def test_merge_preserves_existing_keys(self, tmp_home):
        path = tmp_home / ".config" / "opencode" / "opencode.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "theme": "dark",
            "provider": {"openai": {"options": {"apiKey": "sk-x"}}},
        }))
        onboard.write_opencode_config(home=tmp_home)
        data = json.loads(path.read_text())
        assert data["theme"] == "dark"
        assert data["provider"]["openai"]["options"]["apiKey"] == "sk-x"
        assert "ppmlx" in data["provider"]

    def test_idempotent_and_dry_run(self, tmp_home):
        onboard.write_opencode_config(home=tmp_home)
        record = onboard.write_opencode_config(home=tmp_home)
        assert not record.changed
        dry = onboard.write_opencode_config(home=tmp_home / "elsewhere", dry_run=True)
        assert dry.changed and dry.dry_run
        assert not (tmp_home / "elsewhere").exists()


# ---------------------------------------------------------------------------
# pi (~/.pi/agent/models.json)
# ---------------------------------------------------------------------------


class TestPiModels:
    def test_writes_provider_entry(self, tmp_home):
        onboard.write_pi_models("http://127.0.0.1:6767/v1", "llama3", home=tmp_home)
        path = tmp_home / ".pi" / "agent" / "models.json"
        data = json.loads(path.read_text())
        entry = data["providers"]["ppmlx"]
        assert entry["baseUrl"] == "http://127.0.0.1:6767/v1"
        assert entry["models"][0]["id"] == "llama3"

    def test_preserves_other_providers(self, tmp_home):
        path = tmp_home / ".pi" / "agent" / "models.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"providers": {"other": {"apiKey": "k"}}}))
        onboard.write_pi_models("http://127.0.0.1:6767/v1", "m", home=tmp_home)
        data = json.loads(path.read_text())
        assert data["providers"]["other"] == {"apiKey": "k"}
        assert "ppmlx" in data["providers"]

    def test_idempotent(self, tmp_home):
        onboard.write_pi_models("http://x", "m", home=tmp_home)
        record = onboard.write_pi_models("http://x", "m", home=tmp_home)
        assert not record.changed


# ---------------------------------------------------------------------------
# Route policy setup
# ---------------------------------------------------------------------------


class TestRoutePolicy:
    def test_writes_example_and_config(self, tmp_home):
        cfg = tmp_home / ".ppmlx" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text('[server]\nhost = "127.0.0.1"\n\n[analytics]\nenabled = false\n')
        records = onboard.write_route_policy(home=tmp_home)
        changed_paths = {str(r.path) for r in records if r.changed}
        routes = tmp_home / ".ppmlx" / "routes.toml"
        assert str(routes) in changed_paths
        # Policy parses via load_policy
        from ppmlx.router import load_policy
        policy = load_policy(str(routes))
        assert policy.version == "1"
        data = tomllib.loads(cfg.read_text())
        assert data["server"]["route_policy"] == str(routes)
        assert data["server"]["host"] == "127.0.0.1"
        assert data["analytics"]["enabled"] is False

    def test_does_not_clobber_existing_routes(self, tmp_home):
        routes = tmp_home / ".ppmlx" / "routes.toml"
        routes.parent.mkdir(parents=True)
        routes.write_text('[routes]\nversion = "9"\n')
        records = onboard.write_route_policy(home=tmp_home)
        assert routes.read_text() == '[routes]\nversion = "9"\n'
        route_records = [r for r in records if r.path == routes]
        assert not route_records[0].changed

    def test_idempotent(self, tmp_home):
        onboard.write_route_policy(home=tmp_home)
        records = onboard.write_route_policy(home=tmp_home)
        assert all(not r.changed for r in records)


# ---------------------------------------------------------------------------
# Wizard orchestration
# ---------------------------------------------------------------------------


class TestWizard:
    def test_configure_all(self, tmp_home):
        result = onboard.configure_harnesses(
            ["claude", "codex", "opencode", "pi"],
            model="llama3",
            base_url="http://127.0.0.1:6767",
            home=tmp_home,
        )
        assert len(result.applied) == 4

    def test_full_flow_idempotent(self, tmp_home):
        onboard.configure_harnesses(["claude", "codex"], home=tmp_home)
        onboard.write_route_policy(home=tmp_home)
        second = onboard.configure_harnesses(["claude", "codex", "pi"], home=tmp_home)
        assert all(not r.changed for r in second.changes)

    def test_unknown_harness_raises(self, tmp_home):
        try:
            onboard.configure_harnesses(["cursor"], home=tmp_home)
        except ValueError as error:
            assert "cursor" in str(error)
        else:
            raise AssertionError("expected ValueError")

    def test_detect_harnesses(self, tmp_home, monkeypatch):
        monkeypatch.setattr(onboard.shutil, "which", lambda name: "/bin/codex" if name == "codex" else None)
        (tmp_home / ".pi").mkdir()
        found = onboard.detect_harnesses(tmp_home)
        assert found["codex"] is True  # binary
        assert found["pi"] is True     # config dir
        assert found["claude"] is False
        assert found["opencode"] is False

    def test_dry_run_touches_nothing(self, tmp_home):
        result = onboard.configure_harnesses(
            ["claude", "codex", "opencode"], home=tmp_home, dry_run=True
        )
        assert all(r.dry_run for r in result.changes)
        assert not any((tmp_home / d).exists() for d in (".claude", ".codex", ".config"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

runner = CliRunner()


class TestCliOnboard:
    def test_onboard_non_interactive(self, tmp_home, monkeypatch):
        result = runner.invoke(
            app,
            ["onboard", "--harness", "claude,codex", "--no-setup-routes"],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_home / ".claude" / "settings.json").exists()
        assert (tmp_home / ".codex" / "config.toml").exists()

    def test_onboard_dry_run(self, tmp_home):
        result = runner.invoke(
            app,
            ["onboard", "--harness", "claude", "--dry-run", "--no-setup-routes"],
        )
        assert result.exit_code == 0, result.output
        assert "DRY-RUN" in result.output
        assert not (tmp_home / ".claude" / "settings.json").exists()

    def test_onboard_invalid_harness(self, tmp_home):
        result = runner.invoke(app, ["onboard", "--harness", "cursor"])
        assert result.exit_code == 1
        assert "cursor" in result.output

    def test_persist_launch_config_unified(self, tmp_home, monkeypatch):
        """launch's persistence helper uses the same writers."""
        import ppmlx.config as config_mod
        monkeypatch.setattr(config_mod.Path, "home", lambda: tmp_home, raising=False)
        from ppmlx.cli import _persist_launch_config
        _persist_launch_config("codex", "http://127.0.0.1:6767", "llama3")
        data = tomllib.loads((tmp_home / ".codex" / "config.toml").read_text())
        assert data["model_providers"]["ppmlx"]["base_url"] == "http://127.0.0.1:6767"

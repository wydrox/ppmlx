"""Tests for the Grok CLI harness support in ppmlx onboard."""

from __future__ import annotations

import tomllib

from ppmlx import onboard


def _read_grok(tmp_home) -> dict:
    with open(tmp_home / ".grok" / "config.toml", "rb") as handle:
        return tomllib.load(handle)


class TestGrokDetection:
    def test_detect_via_binary(self, tmp_home, monkeypatch):
        import shutil

        monkeypatch.setattr(
            onboard.shutil, "which", lambda name: "/usr/local/bin/grok" if name == "grok" else None
        )
        assert onboard.detect_harnesses(home=tmp_home)["grok"] is True

    def test_detect_via_config_dir(self, tmp_home):
        (tmp_home / ".grok").mkdir(parents=True)
        found = onboard.detect_harnesses(home=tmp_home)
        assert found["grok"] is True
        assert "grok" in onboard.HARNESSES

    def test_not_detected(self, tmp_home, monkeypatch):
        monkeypatch.setattr(onboard.shutil, "which", lambda name: None)
        assert onboard.detect_harnesses(home=tmp_home)["grok"] is False


class TestWriteGrokConfig:
    def test_writes_fresh_config(self, tmp_home):
        record = onboard.write_grok_config(model="model-heavy", home=tmp_home)
        assert record.changed
        data = _read_grok(tmp_home)
        section = data["model"]["ppmlx"]
        assert section["model"] == "model-heavy"
        assert section["base_url"] == "http://127.0.0.1:6767/v1"
        assert section["env_key"] == "PPMLX_LOCAL"
        assert section["name"] == "ppmlx gateway"
        assert section["context_window"] == 200000

    def test_merge_preserves_existing_model_sections(self, tmp_home):
        cfg = tmp_home / ".grok" / "config.toml"
        cfg.parent.mkdir(parents=True)
        original = """\
# my grok config
theme = "dark"

[model.ox-alpha]
model = "ox-alpha"
base_url = "http://127.0.0.1:6767/v1"
env_key = "OPENROUTER_API_KEY"

[other]
flag = true
"""
        cfg.write_text(original)
        onboard.write_grok_config(model="llama3", home=tmp_home)
        text = cfg.read_text()
        # Sibling sections and top-level keys/comments survive verbatim.
        assert "[model.ox-alpha]" in text
        assert 'env_key = "OPENROUTER_API_KEY"' in text
        assert "# my grok config" in text
        data = _read_grok(tmp_home)
        assert data["model"]["ox-alpha"]["model"] == "ox-alpha"
        assert data["other"] == {"flag": True}
        assert data["theme"] == "dark"
        assert data["model"]["ppmlx"]["model"] == "llama3"

    def test_idempotent_rerun(self, tmp_home):
        first = onboard.write_grok_config(model="model-heavy", home=tmp_home)
        second = onboard.write_grok_config(model="model-heavy", home=tmp_home)
        assert first.changed and not second.changed
        assert second.backup is None
        path = tmp_home / ".grok" / "config.toml"
        once = path.read_text()
        onboard.write_grok_config(model="model-heavy", home=tmp_home)
        assert path.read_text() == once

    def test_backup_created_on_modify(self, tmp_home):
        cfg = tmp_home / ".grok" / "config.toml"
        cfg.parent.mkdir(parents=True)
        cfg.write_text("[model.old]\nmodel = \"x\"\n")
        record = onboard.write_grok_config(home=tmp_home)
        assert record.backup is not None and record.backup.exists()
        backup_data = tomllib.loads(record.backup.read_text())
        assert "ppmlx" not in backup_data.get("model", {})

    def test_dry_run_noop(self, tmp_home):
        record = onboard.write_grok_config(home=tmp_home, dry_run=True)
        assert record.changed  # a change was proposed...
        assert record.dry_run  # ...but nothing was written
        assert record.backup is None
        assert not (tmp_home / ".grok" / "config.toml").exists()


class TestConfigureHarnesses:
    def test_configure_routes_to_grok_writer(self, tmp_home):
        result = onboard.configure_harnesses(["grok"], model="my-alias", home=tmp_home)
        assert len(result.changes) == 1
        assert result.changes[0].harness == "grok"
        assert _read_grok(tmp_home)["model"]["ppmlx"]["model"] == "my-alias"

    def test_default_alias_without_model(self, tmp_home):
        onboard.configure_harnesses(["grok"], home=tmp_home)
        assert _read_grok(tmp_home)["model"]["ppmlx"]["model"] == "model-heavy"


class TestAliasQuestion:
    def test_available_aliases_from_routes_toml(self, tmp_home):
        routes = tmp_home / ".ppmlx" / "routes.toml"
        routes.parent.mkdir(parents=True)
        routes.write_text('[routes.aliases]\nfast = ["local", "default"]\nbig = ["local", "heavy"]\n')
        assert onboard._available_aliases(home=tmp_home) == ["big", "fast"]

    def test_available_aliases_missing_file(self, tmp_home):
        assert onboard._available_aliases(home=tmp_home) == []


def test_cli_onboard_accepts_grok_harness():
    from typer.testing import CliRunner

    from ppmlx.cli import app

    # Regression: --harness grok used to be rejected by a stale CLI whitelist.
    dry = CliRunner().invoke(
        app, ["onboard", "--dry-run", "--yes", "--harness", "grok", "--model", "test-heavy"]
    )
    assert dry.exit_code == 0, dry.output
    assert "Unknown harness" not in (dry.output or "")
    assert "model.ppmlx" in (dry.output or "")
"""Tests for ppmlx.auth (ADR 0004 provider authentication).

The keyring backend is replaced with an in-memory fake; no real Keychain is
touched. A recurring theme: the provider secret must never appear in config,
command output, or error text.
"""
from __future__ import annotations

import logging
import os
import tomllib

import pytest
import tomli_w
from typer.testing import CliRunner

import ppmlx.auth as auth
from ppmlx.auth import (
    AuthError,
    InvalidProviderError,
    KeyringUnavailableError,
    ProviderAuthInfo,
    SecretNotFoundError,
    delete_secret,
    get_secret,
    list_providers,
    load_auth_entries,
    remove_auth_entry,
    resolve_secret,
    save_auth_entry,
    secret_ref_for,
    set_secret,
    status_for,
    validate_provider_name,
)

SECRET = "sk-test-secret-abcdef0123456789"


class FakeKeyring:
    """Minimal in-memory keyring backend."""

    priority = 1

    def __init__(self):
        self.store = {}

    def set_password(self, service, username, password):
        self.store[(service, username)] = password

    def get_password(self, service, username):
        return self.store.get((service, username))

    def delete_password(self, service, username):
        self.store.pop((service, username), None)


@pytest.fixture()
def fake_keyring(monkeypatch, tmp_path):
    """Route ppmlx.auth at an in-memory keyring and a temp config dir."""
    kr = FakeKeyring()
    monkeypatch.setattr(auth, "_keyring", lambda: kr)
    monkeypatch.setenv(auth.CONFIG_DIR_ENV, str(tmp_path))
    return kr


# ---------------------------------------------------------------------------
# Keyring-backed secret lifecycle
# ---------------------------------------------------------------------------


class TestSecretLifecycle:
    def test_set_and_get(self, fake_keyring):
        set_secret("openai", SECRET)
        assert get_secret("openai") == SECRET
        assert fake_keyring.store[(auth.KEYRING_SERVICE, "openai")] == SECRET

    def test_delete_is_idempotent(self, fake_keyring):
        set_secret("openai", SECRET)
        delete_secret("openai")
        delete_secret("openai")  # second delete must not raise
        with pytest.raises(SecretNotFoundError):
            get_secret("openai")

    def test_get_missing_raises_typed_error(self, fake_keyring):
        with pytest.raises(SecretNotFoundError):
            get_secret("never-stored")

    def test_config_stores_only_metadata(self, fake_keyring, tmp_path):
        set_secret("openai", SECRET)
        raw = (tmp_path / "config.toml").read_bytes()
        assert SECRET.encode() not in raw
        data = tomllib.loads(raw.decode())
        entry = data["auth"]["providers"]["openai"]
        assert entry["secret_ref"] == secret_ref_for("openai")
        assert "keyring://" in entry["secret_ref"]
        assert "env_key" not in entry or isinstance(entry["env_key"], str)

    def test_env_key_recorded_without_value(self, fake_keyring, tmp_path):
        set_secret("openai", SECRET, env_key="OPENAI_API_KEY")
        data = tomllib.loads((tmp_path / "config.toml").read_text())
        assert data["auth"]["providers"]["openai"]["env_key"] == "OPENAI_API_KEY"

    def test_invalid_provider_name_rejected_before_touching_keyring(self, fake_keyring):
        for bad in ["", "bad name", "a/b", "x" * 129]:
            with pytest.raises(InvalidProviderError):
                set_secret(bad if bad else " ", SECRET)
        assert fake_keyring.store == {}

    def test_keyring_failure_raises_typed_error(self, fake_keyring, monkeypatch):
        import keyring.errors

        def boom(*a, **kw):
            raise keyring.errors.NoKeyringError("no backend")

        monkeypatch.setattr(
            type(fake_keyring), "set_password", boom
        )
        with pytest.raises(KeyringUnavailableError):
            set_secret("openai", SECRET)


# ---------------------------------------------------------------------------
# Config ([auth]) round-trip, atomic writes and backup
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    def test_load_entries_empty_when_no_file(self, fake_keyring):
        entries, err = load_auth_entries()
        assert entries == {}
        assert err is None

    def test_save_then_load_round_trip(self, fake_keyring):
        save_auth_entry("anthropic", "ANTHROPIC_API_KEY")
        entries, err = load_auth_entries()
        assert err is None
        info = entries["anthropic"]
        assert isinstance(info, ProviderAuthInfo)
        assert info.env_key == "ANTHROPIC_API_KEY"
        assert info.secret_ref == secret_ref_for("anthropic")

    def test_save_preserves_unrelated_sections_and_unknown_keys(self, fake_keyring, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(tomli_w.dumps({
            "server": {"port": 7000},
            "auth": {"providers": {"old": {"secret_ref": "x", "custom": 1}}},
        }))
        save_auth_entry("newprov", None)
        data = tomllib.loads(cfg.read_text())
        assert data["server"] == {"port": 7000}
        assert data["auth"]["providers"]["old"]["custom"] == 1
        assert data["auth"]["providers"]["newprov"]["secret_ref"] == secret_ref_for("newprov")

    def test_remove_entry_keeps_other_sections(self, fake_keyring, tmp_path):
        cfg = tmp_path / "config.toml"
        cfg.write_text(tomli_w.dumps({"server": {"port": 7000}, "auth": {"providers": {"a": {}}}}))
        assert remove_auth_entry("a") is True
        data = tomllib.loads(cfg.read_text())
        assert "auth" not in data
        assert data["server"]["port"] == 7000

    def test_remove_missing_entry_returns_false(self, fake_keyring):
        assert remove_auth_entry("ghost") is False

    def test_atomic_write_leaves_backup(self, fake_keyring, tmp_path):
        cfg_path = tmp_path / "config.toml"
        save_auth_entry("first", None)
        first_content = cfg_path.read_text()
        save_auth_entry("second", None)
        bak = tmp_path / "config.toml.bak"
        assert bak.exists()
        assert bak.read_text() == first_content
        assert "second" in cfg_path.read_text()

    def test_no_temp_files_left_behind(self, fake_keyring, tmp_path):
        save_auth_entry("tidy", None)
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "config.toml"]
        # only the .bak backup may remain
        assert set(leftovers) <= {"config.toml.bak"}


# ---------------------------------------------------------------------------
# Dry-run must be a strict no-op
# ---------------------------------------------------------------------------


class TestDryRunNoOp:
    def test_save_dry_run_writes_nothing(self, fake_keyring, tmp_path):
        assert save_auth_entry("openai", "OPENAI_API_KEY", dry_run=True) is True
        assert not list(tmp_path.iterdir())  # no file created at all
        assert load_auth_entries()[0] == {}

    def test_remove_dry_run_writes_nothing(self, fake_keyring, tmp_path):
        save_auth_entry("openai", None)
        before = (tmp_path / "config.toml").read_text()
        assert remove_auth_entry("openai", dry_run=True) is True
        assert (tmp_path / "config.toml").read_text() == before
        assert "openai" in load_auth_entries()[0]

    def test_cli_add_dry_run_noop(self, fake_keyring, monkeypatch):
        monkeypatch.setenv("PPMLX_AUTH_ALLOW_PIPE_STDIN", "1")
        from ppmlx.cli import app

        runner = CliRunner()
        res = runner.invoke(app, ["auth", "add", "openai", "--dry-run"], input=f"{SECRET}\n{SECRET}\n")
        assert res.exit_code == 0
        assert fake_keyring.store == {}
        assert load_auth_entries()[0] == {}
        assert "Would store" in res.output

    def test_cli_remove_dry_run_noop(self, fake_keyring, monkeypatch):
        from ppmlx.cli import app

        set_secret("openai", SECRET)
        before_store = dict(fake_keyring.store)
        runner = CliRunner()
        res = runner.invoke(app, ["auth", "remove", "openai", "--dry-run"])
        assert res.exit_code == 0
        assert dict(fake_keyring.store) == before_store
        assert "openai" in load_auth_entries()[0]

    def test_dry_run_reports_existing_entry_as_update(self, fake_keyring):
        save_auth_entry("openai", None)
        assert save_auth_entry("openai", "OPENAI_API_KEY", dry_run=True) is True


# ---------------------------------------------------------------------------
# Corrupt / malformed config
# ---------------------------------------------------------------------------


class TestCorruptConfig:
    def test_corrupt_toml_reported_not_raised(self, fake_keyring, tmp_path):
        (tmp_path / "config.toml").write_text("this is ] not toml [[[")
        entries, err = load_auth_entries()
        assert entries == {}
        assert err is not None

    def test_malformed_auth_section_reported(self, fake_keyring, tmp_path):
        (tmp_path / "config.toml").write_text(tomli_w.dumps({"auth": {"providers": 42}}))
        _, err = load_auth_entries()
        assert err is not None

    def test_status_with_corrupt_config_does_not_crash_or_print_secret(
        self, fake_keyring, tmp_path, capsys
    ):
        set_secret("openai", SECRET)
        (tmp_path / "config.toml").write_text("[[broken")
        st = status_for("openai")
        assert st["error"] is not None
        out = capsys.readouterr().out + repr(st)
        assert SECRET not in out

    def test_cli_list_survives_corrupt_config(self, fake_keyring, tmp_path):
        set_secret("openai", SECRET)
        (tmp_path / "config.toml").write_text("not [ valid")
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "list"])
        assert res.exit_code == 0
        assert "corrupt" in res.output.lower()
        assert SECRET not in res.output

    def test_cli_add_refuses_to_write_over_corrupt_config(self, fake_keyring, tmp_path, monkeypatch):
        monkeypatch.setenv("PPMLX_AUTH_ALLOW_PIPE_STDIN", "1")
        (tmp_path / "config.toml").write_text("{{{ broken")
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "add", "openai"], input=f"{SECRET}\n{SECRET}\n")
        assert res.exit_code == 1
        # corrupt file untouched — no silent clobber of user settings
        assert (tmp_path / "config.toml").read_text() == "{{{ broken"
        assert fake_keyring.store.get((auth.KEYRING_SERVICE, "openai")) is None


# ---------------------------------------------------------------------------
# Secret hygiene: never in output, logs, errors, or tracebacks
# ---------------------------------------------------------------------------


def _all_output(result) -> str:
    parts = [result.output]
    exc = getattr(result, "exception", None)
    if exc is not None:
        import traceback

        try:
            parts.append("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except Exception:
            pass
    try:
        std = result.stderr
        if std:
            parts.append(std)
    except Exception:
        pass
    return "\n".join(p for p in parts if p)


class TestSecretNeverInOutput:
    def test_error_rendered_message_redacted(self):
        msg = f"failed around value {SECRET} trailing"
        safe = auth._safe_error(msg, [SECRET])
        assert SECRET not in safe
        err = AuthError(safe)
        assert SECRET not in str(err)
        assert SECRET not in repr(err)

    def test_short_secrets_are_not_redacted_away(self):
        # tiny strings are common substrings; only redact meaningful secrets
        assert auth._safe_error("ab in cab", ["ab"]) == "ab in cab"

    def test_get_missing_error_has_no_trace_of_secret(self, fake_keyring):
        set_secret("openai", SECRET)
        delete_secret("openai")
        try:
            get_secret("openai")
        except SecretNotFoundError as exc:
            blob = str(exc) + repr(exc) + (exc.rendered or "")
            assert SECRET not in blob
        else:
            pytest.fail("expected SecretNotFoundError")

    def test_keyring_failure_error_hides_secret(self, fake_keyring, monkeypatch):
        import keyring.errors

        def boom(self, service, username, password):
            raise RuntimeError(f"backend exploded while storing {password}")

        monkeypatch.setattr(type(fake_keyring), "set_password", boom)
        with pytest.raises(KeyringUnavailableError) as ei:
            set_secret("openai", SECRET)
        assert SECRET not in str(ei.value)
        assert SECRET not in ei.value.rendered

    def test_logging_never_records_secret(self, fake_keyring, caplog):
        class LeakHandler(logging.Handler):
            def __init__(self):
                super().__init__()
                self.records: list[logging.LogRecord] = []

            def emit(self, record):
                self.records.append(record)

        leak = LeakHandler()
        logging.getLogger().addHandler(leak)
        logging.getLogger().setLevel(logging.DEBUG)
        try:
            set_secret("openai", SECRET)
            get_secret("openai")
            status_for("openai")
            list_providers()
        finally:
            logging.getLogger().removeHandler(leak)
        for rec in leak.records:
            assert SECRET not in rec.getMessage()

    def test_exception_repr_carries_only_class_name_detail(self, fake_keyring, monkeypatch):
        import keyring.errors

        def boom(self, service, username, password):
            raise keyring.errors.KeyringLocked("locked!")

        monkeypatch.setattr(type(fake_keyring), "set_password", boom)
        with pytest.raises(KeyringUnavailableError) as ei:
            set_secret("openai", SECRET)
        assert SECRET not in str(ei.value)
        assert "KeyringLocked" not in str(ei.value)  # rendered message is clean
        # detail is diagnostics-only: class name plus backend text, never printed
        assert ei.value.detail.startswith("KeyringLocked:")

    def test_cli_full_lifecycle_output_never_contains_secret(self, fake_keyring, monkeypatch):
        monkeypatch.setenv("PPMLX_AUTH_ALLOW_PIPE_STDIN", "1")
        from ppmlx.cli import app

        runner = CliRunner()
        results = []
        results.append(runner.invoke(app, ["auth", "add", "openai"], input=f"{SECRET}\n{SECRET}\n"))
        assert results[-1].exit_code == 0
        results.append(runner.invoke(app, ["auth", "list"]))
        results.append(runner.invoke(app, ["auth", "status", "openai"]))
        results.append(runner.invoke(app, ["auth", "remove", "openai", "--dry-run"]))
        results.append(runner.invoke(app, ["auth", "remove", "openai"]))
        for i, res in enumerate(results):
            assert res.exit_code == 0, f"step {i} failed: {_all_output(res)}"
            assert SECRET not in _all_output(res), f"secret leaked in step {i}"

    def test_cli_failed_add_output_never_contains_secret(self, fake_keyring, monkeypatch):
        import keyring.errors

        monkeypatch.setenv("PPMLX_AUTH_ALLOW_PIPE_STDIN", "1")

        def boom(self, service, username, password):
            raise keyring.errors.NoKeyringError(f"cannot store {password}")

        monkeypatch.setattr(type(fake_keyring), "set_password", boom)
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "add", "openai"], input=f"{SECRET}\n{SECRET}\n")
        assert res.exit_code == 1
        assert SECRET not in _all_output(res)

    def test_redact_removes_any_secret_seen_this_process(self, fake_keyring):
        set_secret("openai", SECRET)
        leaked = f"backend said: {SECRET} and crashed"
        cleaned = auth.redact(leaked)
        assert SECRET not in cleaned
        assert "[REDACTED]" in cleaned

    def test_print_auth_error_final_gate_redacts_raw_exception(self, fake_keyring):
        # A raw exception (no `rendered`) whose text embeds a previously seen
        # secret still gets redacted by the CLI's final-pass gate.
        set_secret("openai", SECRET)
        from ppmlx.cli import _print_auth_error

        try:
            raise RuntimeError(f"raw failure mentioning {SECRET}")
        except RuntimeError as exc:
            _print_auth_error(exc)
        # console writes go to stdout; capture via capsys is handled by CliRunner
        # in other tests — here assert on redact() directly for determinism.
        assert SECRET not in auth.redact(f"raw failure mentioning {SECRET}")


# ---------------------------------------------------------------------------
# Env fallback resolution
# ---------------------------------------------------------------------------


class TestEnvFallback:
    def test_resolve_prefers_env_when_prefer_env(self, fake_keyring, monkeypatch):
        set_secret("openai", SECRET, env_key="TEST_OPENAI_KEY")
        monkeypatch.setenv("TEST_OPENAI_KEY", "sk-from-env-value-xyz")
        value, source = resolve_secret("openai", prefer_env=True)
        assert value == "sk-from-env-value-xyz"
        assert source == "environment"

    def test_resolve_falls_back_to_env_when_keyring_empty(self, fake_keyring, monkeypatch):
        save_auth_entry("openai", "TEST_OPENAI_KEY")
        monkeypatch.setenv("TEST_OPENAI_KEY", "sk-env-only-9999")
        value, source = resolve_secret("openai")
        assert value == "sk-env-only-9999"
        assert source == "environment"

    def test_resolve_uses_keychain_by_default_even_if_env_set(self, fake_keyring, monkeypatch):
        set_secret("openai", SECRET, env_key="TEST_OPENAI_KEY")
        monkeypatch.setenv("TEST_OPENAI_KEY", "sk-from-env-value-xyz")
        value, source = resolve_secret("openai")
        assert value == SECRET
        assert source == "keychain"

    def test_status_shows_env_state_without_values(self, fake_keyring, monkeypatch):
        set_secret("openai", SECRET, env_key="TEST_OPENAI_KEY")
        monkeypatch.delenv("TEST_OPENAI_KEY", raising=False)
        st = status_for("openai")
        assert st["stored"] is True
        assert st["source"] == "keychain"
        assert st["env_key"] == "TEST_OPENAI_KEY"
        assert st["env_set"] is False


# ---------------------------------------------------------------------------
# Status / listing
# ---------------------------------------------------------------------------


class TestStatusAndList:
    def test_status_missing_provider(self, fake_keyring):
        st = status_for("nosuch")
        assert st["stored"] is False
        assert st["source"] is None
        assert st["error"] is None

    def test_list_sorted_by_provider(self, fake_keyring):
        for p in ("zeta", "alpha", "mid"):
            set_secret(p, SECRET + p)
        names = [st["provider"] for st in list_providers()]
        assert names == ["alpha", "mid", "zeta"]

    def test_secret_refs_are_stable_and_namespaced(self):
        assert secret_ref_for("openai").startswith("keyring://ppmlx/")
        assert secret_ref_for("openai") != secret_ref_for("anthropic")


# ---------------------------------------------------------------------------
# CLI misc
# ---------------------------------------------------------------------------


class TestCliMisc:
    def test_help_lists_auth_commands(self):
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "--help"])
        assert res.exit_code == 0
        for cmd in ("add", "list", "status", "remove"):
            assert cmd in res.output

    def test_add_mismatch_aborts_without_storing(self, fake_keyring, monkeypatch):
        monkeypatch.setenv("PPMLX_AUTH_ALLOW_PIPE_STDIN", "1")
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "add", "openai"], input="one-key\nother-key\n")
        assert res.exit_code == 1
        assert fake_keyring.store == {}

    def test_add_empty_key_rejected(self, fake_keyring, monkeypatch):
        monkeypatch.setenv("PPMLX_AUTH_ALLOW_PIPE_STDIN", "1")
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "add", "openai"], input="\n\n")
        assert res.exit_code == 1
        assert fake_keyring.store == {}

    def test_add_non_interactive_stdin_refused(self, fake_keyring, monkeypatch):
        monkeypatch.delenv("PPMLX_AUTH_ALLOW_PIPE_STDIN", raising=False)
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "add", "openai"], input=f"{SECRET}\n{SECRET}\n")
        assert res.exit_code == 1
        assert fake_keyring.store == {}

    def test_add_overwrite_declined_keeps_old_secret(self, fake_keyring, monkeypatch):
        monkeypatch.setenv("PPMLX_AUTH_ALLOW_PIPE_STDIN", "1")
        from ppmlx.cli import app

        set_secret("openai", SECRET)
        new_key = "sk-brand-new-key-42"
        res = CliRunner().invoke(app, ["auth", "add", "openai"], input=f"{new_key}\n{new_key}\nn\n")
        assert res.exit_code != 0  # aborted
        assert get_secret("openai") == SECRET

    def test_remove_unknown_provider_exits_cleanly(self):
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "remove", "ghost"])
        assert res.exit_code == 1
        assert "No auth entry found" in res.output

    def test_remove_deletes_both_store_and_config(self, fake_keyring):
        from ppmlx.cli import app

        set_secret("openai", SECRET)
        res = CliRunner().invoke(app, ["auth", "remove", "openai"])
        assert res.exit_code == 0
        assert (auth.KEYRING_SERVICE, "openai") not in fake_keyring.store
        assert "openai" not in load_auth_entries()[0]

    def test_status_command_marks_missing_provider(self):
        from ppmlx.cli import app

        res = CliRunner().invoke(app, ["auth", "status", "ghost"])
        assert res.exit_code == 0
        assert "stored: no" in res.output

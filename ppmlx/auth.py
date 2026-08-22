"""Provider authentication storage (ADR 0004).

Secrets live in the operating system protected store — the macOS Keychain via
the ``keyring`` library. The ppmlx config file (``~/.ppmlx/config.toml``) only
ever stores non-secret metadata for each provider entry under ``[auth]``:

.. code-block:: toml

    [auth.providers.openai]
    env_key = "OPENAI_API_KEY"          # optional environment fallback name
    secret_ref = "keyring://ppmlx/openai"

The secret value itself never touches disk. Error messages produced by this
module are rendered through :func:`_safe_error`, which strips any text that
matches the secret values seen during the operation.
"""
from __future__ import annotations

import os
import tempfile
import tomllib
import tomli_w
from dataclasses import dataclass
from pathlib import Path
from typing import Any

KEYRING_SERVICE = "ppmlx"
SECRET_REF_PREFIX = f"keyring://{KEYRING_SERVICE}/"
CONFIG_DIR_ENV = "PPMLX_CONFIG_DIR"


class AuthError(Exception):
    """Base class for provider-auth errors.

    ``rendered`` carries a message that is safe to print or log: every known
    secret value has been removed from it. Subclasses must always populate it
    via ``_safe_error()`` so secrets cannot leak through str(exc) or logs.
    """

    def __init__(self, rendered: str, *, detail: str = "") -> None:
        super().__init__(rendered)
        self.rendered = rendered
        self.detail = detail

    def __str__(self) -> str:
        return self.rendered


class ProviderNotFoundError(AuthError):
    """No auth profile is stored for the requested provider."""


class SecretNotFoundError(AuthError):
    """A profile exists but its keyring secret is missing."""


class KeyringUnavailableError(AuthError):
    """The system keyring backend could not be used."""


class InvalidProviderError(AuthError):
    """The provider identifier is not a usable account name."""


def get_config_dir() -> Path:
    """Return the ppmlx config directory (overridable for tests)."""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / ".ppmlx"


def _config_path(config_dir: Path | None = None) -> Path:
    return (config_dir or get_config_dir()) / "config.toml"


def validate_provider_name(provider: str) -> str:
    """Validate a provider id and return the normalized form."""
    normalized = (provider or "").strip().lower()
    if not normalized:
        raise InvalidProviderError(_safe_error("Provider name must not be empty.", []))
    if not all(c.isalnum() or c in "-_" for c in normalized):
        raise InvalidProviderError(
            _safe_error(
                f"Invalid provider name {normalized!r}: use letters, digits, '-' or '_'.",
                [],
            )
        )
    if len(normalized) > 128:
        raise InvalidProviderError(
            _safe_error("Provider name is too long (max 128 characters).", [])
        )
    return normalized


def secret_ref_for(provider: str) -> str:
    """Return the canonical secret_ref for a provider id."""
    return SECRET_REF_PREFIX + provider


def _provider_from_secret_ref(secret_ref: str) -> str | None:
    if isinstance(secret_ref, str) and secret_ref.startswith(SECRET_REF_PREFIX):
        rest = secret_ref[len(SECRET_REF_PREFIX):]
        if rest and "/" not in rest:
            return rest
    return None


# ---------------------------------------------------------------------------
# Config ([auth]) handling — metadata only; secrets stay in the keyring.
# ---------------------------------------------------------------------------


@dataclass
class ProviderAuthInfo:
    """Non-secret view of one provider's auth entry."""

    provider: str
    env_key: str | None
    secret_ref: str

    @property
    def backend(self) -> str:
        return "env" if self.env_key else "keyring"


def load_auth_config(config_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    """Load raw [auth] entries from config.toml.

    A corrupt file raises :class:`tomllib.TOMLDecodeError` to the caller;
    use :func:`load_auth_entries` for a CLI-safe variant.
    """
    path = _config_path(config_dir)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return {}
    auth = data.get("auth")
    if not isinstance(auth, dict):
        return {}
    providers = auth.get("providers")
    if not isinstance(providers, dict):
        return {}
    return {str(k): v for k, v in providers.items() if isinstance(v, dict)}


def load_auth_entries(
    config_dir: Path | None = None,
) -> tuple[dict[str, ProviderAuthInfo], Exception | None]:
    """Load parsed auth entries.

    Returns ``(entries, error)`` where ``error`` is set when config.toml is
    corrupt or the [auth] section is malformed. Callers can still list other
    sections' state but MUST NOT write until the user resolves the conflict,
    otherwise unrelated settings would be lost.
    """
    path = _config_path(config_dir)
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        return {}, None
    except tomllib.TOMLDecodeError as exc:
        return {}, exc
    auth = data.get("auth")
    entries: dict[str, ProviderAuthInfo] = {}
    if auth is None:
        return entries, None
    if not isinstance(auth, dict):
        return entries, ValueError("[auth] must be a table")
    providers = auth.get("providers", {})
    if not isinstance(providers, dict):
        return entries, ValueError("[auth.providers] must be a table")
    for name, raw in providers.items():
        if not isinstance(raw, dict):
            continue
        entries[str(name)] = ProviderAuthInfo(
            provider=str(name),
            env_key=str(raw["env_key"]) if raw.get("env_key") else None,
            secret_ref=secret_ref_for(str(name)),
        )
    return entries, None


def save_auth_entry(
    provider: str,
    env_key: str | None,
    *,
    dry_run: bool = False,
    config_dir: Path | None = None,
) -> bool:
    """Persist one provider's non-secret auth metadata atomically.

    Writes go to a temp file in the same directory followed by an atomic
    ``os.replace``; the previous file is kept as ``config.toml.bak``.

    Returns True when a change was written (or would be written), False when
    there was nothing to change. In ``dry_run`` mode nothing is written.
    """
    provider = validate_provider_name(provider)
    path = _config_path(config_dir)
    try:
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    except FileNotFoundError:
        data = {}

    providers = data.setdefault("auth", {}).setdefault("providers", {})
    existing = providers.get(provider)
    new_entry: dict[str, Any] = {"secret_ref": secret_ref_for(provider)}
    if env_key:
        new_entry["env_key"] = env_key
    if isinstance(existing, dict) and existing == new_entry:
        return False
    if isinstance(existing, dict):
        # Preserve unknown keys on update so we do not drop foreign metadata.
        merged = dict(existing)
        merged.update(new_entry)
        new_entry = merged
    providers[provider] = new_entry

    if dry_run:
        return True
    _atomic_write_toml(path, data)
    return True


def remove_auth_entry(
    provider: str,
    *,
    dry_run: bool = False,
    config_dir: Path | None = None,
) -> bool:
    """Remove one provider's auth metadata atomically.

    Returns True when the entry existed (was removed / would be removed).
    """
    provider = validate_provider_name(provider)
    path = _config_path(config_dir)
    try:
        with open(path, "rb") as f:
            data: dict[str, Any] = tomllib.load(f)
    except FileNotFoundError:
        return False
    auth = data.get("auth")
    if not isinstance(auth, dict):
        return False
    providers = auth.get("providers")
    if not isinstance(providers, dict) or provider not in providers:
        return False
    del providers[provider]
    if not providers:
        del auth["providers"]
    if not auth:
        del data["auth"]
    if dry_run:
        return True
    _atomic_write_toml(path, data)
    return True


def _atomic_write_toml(path: Path, data: dict[str, Any]) -> None:
    """Write TOML atomically, keeping a .bak of the previous content."""
    payload = tomli_w.dumps(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".config-", suffix=".toml", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.replace(path, path.with_suffix(".toml.bak"))
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Keyring operations
# ---------------------------------------------------------------------------

try:  # pragma: no cover - trivial re-export shim for tests/mocking
    import keyring
except ImportError as _exc:  # pragma: no cover - keyring is a hard dependency
    raise ImportError(
        "The 'keyring' package is required for provider authentication storage"
    ) from _exc


def _keyring() -> Any:
    return keyring


def _backend_name(kr: Any) -> str:
    try:
        return type(kr.get_keyring()).__name__
    except Exception:
        return "<unknown>"


def _safe_error(message: str, secrets: list[str]) -> str:
    """Redact every known secret occurrence from an error message.

    Non-empty secret values are replaced with ``[REDACTED]`` before the text
    is shown or logged anywhere.
    """
    safe = message
    for s in secrets:
        if s and len(s) >= 3 and s in safe:
            safe = safe.replace(s, "[REDACTED]")
    return safe


# Every secret value this process has touched, kept so display code can apply
# a final-pass redaction regardless of how a message was constructed.
_seen_secrets: list[str] = []


def _remember_secret(value: str | None) -> None:
    if value and len(value) >= 3 and value not in _seen_secrets:
        _seen_secrets.append(value)


def redact(text: str) -> str:
    """Remove any secret seen by this process from ``text``."""
    for s in tuple(_seen_secrets):
        if s in text:
            text = text.replace(s, "[REDACTED]")
    return text


def set_secret(
    provider: str,
    secret: str,
    *,
    env_key: str | None = None,
    config_dir: Path | None = None,
) -> None:
    """Store a provider API key in the OS protected store (macOS Keychain).

    ``env_key`` records (in config.toml only) the name of an environment
    variable that takes precedence over the keychain entry when set.
    """
    provider = validate_provider_name(provider)
    _remember_secret(secret)
    kr = _keyring()
    try:
        kr.set_password(KEYRING_SERVICE, provider, secret)
    except keyring.errors.NoKeyringError as exc:
        raise KeyringUnavailableError(
            _safe_error("System keyring is not available.", [secret]),
            detail=f"{type(exc).__name__}: {exc}",
        ) from None
    except keyring.errors.KeyringLocked as exc:
        raise KeyringUnavailableError(
            _safe_error("System keyring is locked; unlock it and retry.", [secret]),
            detail=f"{type(exc).__name__}: {exc}",
        ) from None
    except Exception as exc:
        raise KeyringUnavailableError(
            _safe_error(f"Failed to store secret in system keyring: {_exc_class(exc)}", [secret]),
            detail=f"{type(exc).__name__}: {exc}",
        ) from None
    save_auth_entry(provider, env_key, config_dir=config_dir)


def get_secret(provider: str) -> str:
    """Retrieve a provider API key from the OS protected store."""
    provider = validate_provider_name(provider)
    kr = _keyring()
    try:
        value = kr.get_password(KEYRING_SERVICE, provider)
    except keyring.errors.NoKeyringError as exc:
        raise KeyringUnavailableError(
            _safe_error("System keyring is not available.", []),
            detail=f"{type(exc).__name__}: {exc}",
        ) from None
    except keyring.errors.KeyringLocked as exc:
        raise KeyringUnavailableError(
            _safe_error("System keyring is locked; unlock it and retry.", []),
            detail=f"{type(exc).__name__}: {exc}",
        ) from None
    except Exception as exc:
        raise KeyringUnavailableError(
            _safe_error(f"Failed to read secret from system keyring: {_exc_class(exc)}", []),
            detail=f"{type(exc).__name__}: {exc}",
        ) from None
    if value is None:
        raise SecretNotFoundError(
            _safe_error(f"No API key stored for provider {provider!r}.", []),
        )
    _remember_secret(value)
    return value


def delete_secret(provider: str) -> None:
    """Delete a provider API key from the OS protected store."""
    provider = validate_provider_name(provider)
    kr = _keyring()
    try:
        kr.delete_password(KEYRING_SERVICE, provider)
    except keyring.errors.PasswordDeleteError:
        # Nothing stored (or already gone) — deleting is idempotent.
        pass
    except (keyring.errors.NoKeyringError, keyring.errors.KeyringLocked) as exc:
        raise KeyringUnavailableError(
            _safe_error("System keyring is not available or locked.", []),
            detail=f"{type(exc).__name__}: {exc}",
        ) from None
    except Exception as exc:
        raise KeyringUnavailableError(
            _safe_error(f"Failed to delete secret from system keyring: {_exc_class(exc)}", []),
            detail=f"{type(exc).__name__}: {exc}",
        ) from None


def resolve_secret(provider: str, *, prefer_env: bool = False) -> tuple[str, str]:
    """Resolve the credential source for a provider without printing it.

    Returns ``(secret_value, source_label)`` where source_label is one of
    ``"environment"`` or ``"keychain"``. When ``prefer_env`` is set the
    environment variable named by the stored profile wins over the keyring.
    """
    entries, err = load_auth_entries()
    env_key = None
    if err is None:
        info = entries.get(provider)
        if info is not None:
            env_key = info.env_key
    if env_key and (prefer_env or not _has_stored_secret(provider)):
        env_val = os.environ.get(env_key, "")
        if env_val:
            _remember_secret(env_val)
            return env_val, "environment"
    secret = get_secret(provider)
    return secret, "keychain"


def status_for(provider: str, *, prefer_env: bool = False) -> dict[str, Any]:
    """Build a printable, secret-free status report for one provider."""
    provider = validate_provider_name(provider)
    entries, err = load_auth_entries()
    info = entries.get(provider)
    result: dict[str, Any] = {
        "provider": provider,
        "stored": False,
        "source": None,
        "env_key": info.env_key if info else None,
        "secret_ref": info.secret_ref if info else secret_ref_for(provider),
        "env_set": False,
        "error": None,
    }
    if err is not None:
        result["error"] = "config.toml is corrupt or unreadable; run `ppmlx doctor` or repair it manually"
        return result
    if info is not None and info.env_key:
        result["env_set"] = bool(os.environ.get(info.env_key))
    if info is not None and info.env_key and (prefer_env or not _has_stored_secret(provider)):
        env_val = os.environ.get(info.env_key)
        if env_val:
            result["stored"] = True
            result["source"] = "environment"
            return result
    try:
        get_secret(provider)
    except SecretNotFoundError:
        pass
    except KeyringUnavailableError as exc:
        result["error"] = exc.rendered
        return result
    else:
        result["stored"] = True
        result["source"] = "keychain"
    return result


def list_providers(*, prefer_env: bool = False) -> list[dict[str, Any]]:
    """List all configured providers with secret-free status information."""
    entries, err = load_auth_entries()
    names = sorted(entries.keys())
    results = []
    for name in names:
        st = status_for(name, prefer_env=prefer_env) if err is None else {
            "provider": name,
            "stored": False,
            "source": None,
            "env_key": entries[name].env_key,
            "secret_ref": entries[name].secret_ref,
            "env_set": False,
            "error": "config.toml is corrupt or unreadable",
        }
        results.append(st)
    return results


def _has_stored_secret(provider: str) -> bool:
    try:
        get_secret(provider)
        return True
    except AuthError:
        return False


def _exc_class(exc: BaseException) -> str:
    return type(exc).__name__


__all__ = [
    "AuthError",
    "InvalidProviderError",
    "KeyringUnavailableError",
    "ProviderNotFoundError",
    "ProviderAuthInfo",
    "SECRET_REF_PREFIX",
    "KEYRING_SERVICE",
    "SecretNotFoundError",
    "delete_secret",
    "get_config_dir",
    "get_secret",
    "list_providers",
    "load_auth_entries",
    "load_auth_config",
    "redact",
    "remove_auth_entry",
    "resolve_secret",
    "save_auth_entry",
    "secret_ref_for",
    "set_secret",
    "status_for",
    "validate_provider_name",
]

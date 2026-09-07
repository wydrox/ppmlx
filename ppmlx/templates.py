"""Prompt template engine: load, validate, and render YAML prompt templates."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


# ── Exceptions ────────────────────────────────────────────────────────────

class TemplateError(Exception):
    """Base error for template operations."""


class TemplateNotFoundError(TemplateError):
    """Raised when a template name cannot be resolved."""


class TemplateValidationError(TemplateError):
    """Raised when a template file is malformed."""


class TemplateMissingVariableError(TemplateError):
    """Raised when a required variable is not supplied."""


# ── Data model ────────────────────────────────────────────────────────────

@dataclass
class VariableSpec:
    required: bool = True
    default: str | None = None
    description: str = ""


@dataclass
class PromptTemplate:
    name: str
    description: str = ""
    system: str = ""
    prompt: str = ""
    variables: dict[str, VariableSpec] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    default_model: str | None = None
    source: str = "built-in"  # "built-in" | "user"


# ── Template directories ─────────────────────────────────────────────────

_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")


def _builtin_dir() -> Path:
    """Return the directory containing built-in YAML templates."""
    return Path(__file__).parent / "templates"


def _user_dir(*, create: bool = False) -> Path:
    """Return the user template directory (~/.ppmlx/templates/).

    Only creates the directory when *create* is True (i.e. write operations).
    """
    from ppmlx.config import get_ppmlx_dir
    d = get_ppmlx_dir() / "templates"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


# ── Parsing ───────────────────────────────────────────────────────────────

def _parse_variable(spec: Any) -> VariableSpec:
    """Parse a variable spec from YAML (dict or plain string)."""
    if isinstance(spec, dict):
        has_default = "default" in spec
        return VariableSpec(
            required=spec.get("required", not has_default),
            default=str(spec["default"]) if has_default else None,
            description=str(spec.get("description", "")),
        )
    return VariableSpec(required=True, description=str(spec) if spec else "")


def load_template_from_yaml(text: str, *, source: str = "built-in") -> PromptTemplate:
    """Parse a YAML string into a PromptTemplate."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise TemplateValidationError(f"Invalid YAML: {exc}") from exc

    if not isinstance(data, dict):
        raise TemplateValidationError("Template YAML must be a mapping (dict)")

    name = data.get("name")
    if not name:
        raise TemplateValidationError("Template must have a 'name' field")

    variables: dict[str, VariableSpec] = {}
    raw_vars = data.get("variables", {})
    if isinstance(raw_vars, dict):
        for var_name, var_spec in raw_vars.items():
            variables[str(var_name)] = _parse_variable(var_spec)

    parameters: dict[str, Any] = {}
    raw_params = data.get("parameters", {})
    if isinstance(raw_params, dict):
        parameters = dict(raw_params)

    return PromptTemplate(
        name=str(name),
        description=str(data.get("description", "")),
        system=str(data.get("system", "")),
        prompt=str(data.get("prompt", "")),
        variables=variables,
        parameters=parameters,
        default_model=str(data["default_model"]) if data.get("default_model") else None,
        source=source,
    )


def load_template_from_file(path: Path, *, source: str = "built-in") -> PromptTemplate:
    """Load a PromptTemplate from a YAML file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateNotFoundError(f"Cannot read template file: {exc}") from exc
    return load_template_from_yaml(text, source=source)


# ── Discovery ─────────────────────────────────────────────────────────────

def _load_from_dir(directory: Path, source: str) -> dict[str, PromptTemplate]:
    """Load all YAML templates from a directory, keyed by name."""
    templates: dict[str, PromptTemplate] = {}
    if directory.is_dir():
        for f in sorted(directory.glob("*.yaml")):
            try:
                t = load_template_from_file(f, source=source)
                templates[t.name] = t
            except TemplateError:
                pass
    return templates


def list_templates() -> list[PromptTemplate]:
    """Return all available templates (built-in + user), user overrides built-in."""
    templates = _load_from_dir(_builtin_dir(), "built-in")
    templates.update(_load_from_dir(_user_dir(), "user"))
    return sorted(templates.values(), key=lambda t: t.name)


def get_template(name: str) -> PromptTemplate:
    """Look up a template by name. User templates take priority."""
    user_file = _user_dir() / f"{name}.yaml"
    if user_file.is_file():
        return load_template_from_file(user_file, source="user")

    builtin_file = _builtin_dir() / f"{name}.yaml"
    if builtin_file.is_file():
        return load_template_from_file(builtin_file, source="built-in")

    raise TemplateNotFoundError(
        f"Template '{name}' not found. Run 'ppmlx template list' to see available templates."
    )


# ── Rendering ─────────────────────────────────────────────────────────────


def render_template(
    template: PromptTemplate,
    variables: dict[str, str],
) -> tuple[str, str]:
    """Render a template with the given variables.

    Returns (system_prompt, user_prompt).
    Raises TemplateMissingVariableError if a required variable is missing.
    """
    # Build the effective variable values: supplied > defaults
    effective: dict[str, str] = {}
    for var_name, spec in template.variables.items():
        if var_name in variables:
            effective[var_name] = variables[var_name]
        elif spec.default is not None:
            effective[var_name] = spec.default
        elif spec.required:
            raise TemplateMissingVariableError(
                f"Required variable '{var_name}' not provided. "
                f"Use --var {var_name}=<value>"
            )

    # Also allow extra variables not declared in the spec
    for k, v in variables.items():
        if k not in effective:
            effective[k] = v

    def _substitute(text: str) -> str:
        def _replace(match: re.Match) -> str:
            key = match.group(1)
            return effective.get(key, match.group(0))
        return _VAR_PATTERN.sub(_replace, text)

    system = _substitute(template.system)
    prompt = _substitute(template.prompt)
    return system, prompt


def read_stdin_if_available() -> str | None:
    """Read from stdin if it's not a terminal (piped data). Returns None if stdin is a terminal."""
    if sys.stdin.isatty():
        return None
    return sys.stdin.read()

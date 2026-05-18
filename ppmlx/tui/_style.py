"""Shared styles and helpers for prompt_toolkit TUI components."""
from __future__ import annotations


def get_style():
    """Return a prompt_toolkit Style that works on both dark and light terminals."""
    from prompt_toolkit.styles import Style

    return Style.from_dict({
        "header": "bold",
        "header.cmd": "bold cyan",
        "cursor": "reverse",
        "section": "bold italic",
        "dim": "#888888",
        "size": "#888888",
        "star": "yellow",
        "footer": "#888888 italic",
        "checked": "bold green",
        "unchecked": "",
        "disabled": "#666666 italic",
        "desc": "#888888",
        "unsaved": "bold yellow",
        "value": "cyan",
        "table.border": "#555555",
        "table.header": "bold",
    })


def version_str() -> str:
    """Return the ppmlx version string."""
    try:
        from importlib.metadata import version
        return version("ppmlx")
    except Exception:
        return "dev"


# ── Table column widths ──────────────────────────────────────────────

COL_W = {
    "cursor":  4,   # "  ▸ " or "    "
    "check":   4,   # "[x] " or "[ ] "
    "flag":    3,   # "★✓●"
    "alias":  42,
    "params":  8,
    "precision":  10,
    "size":   10,
    "downloads":  10,
    "updated":  12,
}


def _clip(text: str, width: int) -> str:
    """Clip text to a fixed terminal column width using a single ellipsis."""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _pad(text: str, width: int, align: str = "left") -> str:
    text = _clip(str(text), width)
    if align == "right":
        return text.rjust(width)
    return text.ljust(width)


# ── Unified table rendering ──────────────────────────────────────────

def render_table_header(*, show_checkbox: bool = False) -> list[tuple[str, str]]:
    """Render column header row + separator line."""
    w = COL_W
    parts: list[str] = []
    parts.append(" " * w["cursor"])
    if show_checkbox:
        parts.append(" " * w["check"])
    parts.append(_pad("", w["flag"]))
    parts.append(" ")
    parts.append(_pad("Alias", w["alias"]))
    parts.append(_pad("Params", w["params"], "right"))
    parts.append(_pad("Precision", w["precision"], "right"))
    parts.append(_pad("Size", w["size"], "right"))
    parts.append(_pad("Downloads", w["downloads"], "right"))
    parts.append(_pad("Updated", w["updated"], "right"))
    header_line = "".join(parts)

    total_w = len(header_line)
    sep = "─" * total_w

    return [
        ("class:table.header", header_line),
        ("", "\n"),
        ("class:table.border", sep),
        ("", "\n"),
    ]


def render_section_title(title: str) -> list[tuple[str, str]]:
    """Render a section title (e.g. 'Downloaded', 'Available') above a table."""
    return [
        ("", "\n"),
        ("class:section", f"  {title}"),
        ("", "\n"),
    ]


def render_model_row(
    row,
    *,
    is_cursor: bool = False,
    checkbox: str | None = None,
) -> list[tuple[str, str]]:
    """Render one table row: cursor | [checkbox] | flags | alias | params | precision | size.

    ``row`` must have: alias, params_b, precision, size_gb, downloads, updated_at, is_favorite, downloaded, is_loaded.
    ``checkbox`` — "[x]" or "[ ]" for multi-select; None to hide the column.
    """
    w = COL_W
    style = "class:cursor" if is_cursor else ""

    # Cursor prefix
    prefix = "  \u25b8 " if is_cursor else "    "

    fragments: list[tuple[str, str]] = []
    fragments.append((style, prefix))

    # Checkbox
    if checkbox is not None:
        cb_style = "class:checked" if checkbox == "[x]" else "class:unchecked"
        fragments.append((cb_style if not is_cursor else style, f"{checkbox} "))

    # Status flags
    flags = ""
    if getattr(row, "is_favorite", False):
        flags += "★"
    if getattr(row, "downloaded", False):
        flags += "✓"
    if getattr(row, "is_loaded", False):
        flags += "●"
    fragments.append(("class:star" if not is_cursor else style, _pad(flags, w["flag"])))
    fragments.append((style, " "))

    # Data columns
    alias = _pad(row.alias, w["alias"])
    params = _pad(f"{row.params_b}B" if row.params_b else "—", w["params"], "right")
    precision = _pad(getattr(row, "precision", None) or "—", w["precision"], "right")
    size = _pad(f"{row.size_gb:.1f} GB" if row.size_gb else "—", w["size"], "right")
    downloads_raw = getattr(row, "downloads", None)
    downloads = _pad(f"{round(downloads_raw / 1000):,}k" if downloads_raw is not None else "—", w["downloads"], "right")
    updated = _pad(getattr(row, "updated_at", None) or "—", w["updated"], "right")

    if is_cursor:
        fragments.extend([(style, alias), (style, params), (style, precision), (style, size), (style, downloads), (style, updated)])
    else:
        fragments.extend([("", alias), ("class:dim", params), ("class:dim", precision), ("class:size", size), ("class:dim", downloads), ("class:dim", updated)])

    fragments.append(("", "\n"))
    return fragments


def header_text(command: str = "ppmlx") -> list[tuple[str, str]]:
    """Return header formatted text tuples."""
    return [
        ("class:header", f"ppmlx {version_str()}"),
        ("", "\n"),
        ("class:header.cmd", f"$ {command}"),
        ("", "\n\n"),
    ]

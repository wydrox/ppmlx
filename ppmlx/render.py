"""Output rendering helpers for the ppmlx REPL and agent.

Provides markdown formatting and generation statistics display,
controlled by [ui] settings in config.toml.
"""
from __future__ import annotations

import time
from typing import Iterator

from rich.console import Console
from rich.markdown import Markdown


def print_response(
    text: str,
    *,
    console: Console,
    markdown: bool = False,
    prefix: str = "",
) -> None:
    """Print an assistant/agent response, optionally rendered as Markdown."""
    if prefix:
        console.print(prefix, end="")

    if markdown and text.strip():
        console.print(Markdown(text))
    else:
        console.print(text)


def stream_and_collect(
    gen: Iterator[str],
    *,
    console: Console,
    markdown: bool = False,
    show_stats: bool = False,
    prefix: str = "\n[bold green]Assistant:[/bold green] ",
) -> tuple[str, dict]:
    """Stream text chunks, collecting them and tracking timing stats.

    Returns (full_text, stats_dict) where stats_dict has:
      ttft_ms, total_ms, tokens, tok_per_sec
    """
    full_text: list[str] = []
    t_start = time.monotonic()
    ttft_ms: float | None = None
    token_count = 0

    if not markdown:
        # Streaming print: show tokens as they arrive
        console.print(prefix, end="")
        for chunk in gen:
            full_text.append(chunk)
            console.print(chunk, end="", highlight=False)
            if ttft_ms is None and chunk.strip():
                ttft_ms = (time.monotonic() - t_start) * 1000
            token_count += 1
        console.print()  # newline after streamed content
    else:
        # Collect first, then render markdown
        console.print(prefix, end="")
        for chunk in gen:
            full_text.append(chunk)
            if ttft_ms is None and chunk.strip():
                ttft_ms = (time.monotonic() - t_start) * 1000
            token_count += 1
        text = "".join(full_text)
        if text.strip():
            console.print(Markdown(text))
        else:
            console.print()

    total_ms = (time.monotonic() - t_start) * 1000
    tok_per_sec = token_count / (total_ms / 1000) if total_ms > 0 else 0.0

    stats = {
        "ttft_ms": round(ttft_ms or 0.0, 1),
        "total_ms": round(total_ms, 1),
        "tokens": token_count,
        "tok_per_sec": round(tok_per_sec, 1),
    }

    if show_stats and token_count > 0:
        _print_stats(console, stats)

    return "".join(full_text), stats


def _print_stats(console: Console, stats: dict) -> None:
    """Print a compact generation stats line."""
    parts = []
    if stats["ttft_ms"] > 0:
        parts.append(f"TTFT {stats['ttft_ms']:.0f}ms")
    parts.append(f"{stats['tok_per_sec']:.1f} tok/s")
    parts.append(f"{stats['tokens']} tokens")
    parts.append(f"{stats['total_ms']/1000:.1f}s")
    console.print(f"[dim]  ·  {' · '.join(parts)}[/dim]")

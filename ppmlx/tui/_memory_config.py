"""Dedicated memory config editor TUI using prompt_toolkit."""
from __future__ import annotations


def memory_config_menu() -> None:
    """Show an interactive editor for [memory] config only."""
    import tomllib

    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    from ppmlx.config import get_ppmlx_dir
    from ppmlx.tui._style import get_style, header_text

    cfg_path = get_ppmlx_dir() / "config.toml"
    data: dict = {}
    try:
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        pass

    memory = data.get("memory", {})
    modes = ["off", "shadow", "compact", "inject"]
    extractors = ["hybrid"]
    enabled = bool(memory.get("enabled", True))
    mode = str(memory.get("mode", "off")).lower()
    if mode not in modes:
        mode = "off"
    extractor = _normalize_extractor(memory.get("extractor", "rule_based"))
    model = str(memory.get("extraction_model", "gemma-4-e2b"))

    candidate_options = [0, 4, 8, 12, 20, 32]
    worker_options = [1, 2, 4, 8]
    output_options = [256, 512, 900, 1200, 2000, 4000]
    input_options = [1024, 2048, 4096, 6000, 8192, 12000]
    overlap_options = [0, 256, 600, 1000, 1600]
    chunk_options = [4, 8, 16, 32, 64, 128]
    timeout_options = [5.0, 15.0, 45.0, 90.0, 180.0]

    state = {
        "cursor": 0,
        "enabled": enabled,
        "mode_index": modes.index(mode),
        "extractor_index": extractors.index(extractor),
        "model": model,
        "candidates_index": _option_index(candidate_options, memory.get("max_candidates_per_event", 12)),
        "workers_index": _option_index(worker_options, memory.get("extraction_workers", 1)),
        "output_index": _option_index(output_options, memory.get("extraction_max_tokens", 1200)),
        "input_index": _option_index(input_options, memory.get("extraction_input_tokens", 6000)),
        "overlap_index": _option_index(overlap_options, memory.get("extraction_overlap_tokens", 600)),
        "chunks_index": _option_index(chunk_options, memory.get("extraction_max_chunks_per_event", 32)),
        "timeout_index": _option_index(timeout_options, memory.get("extraction_timeout_seconds", 45.0)),
        "editing_model": False,
        "model_buf": "",
        "dirty": False,
        "saved_flash": False,
    }
    items = [
        "enabled", "mode", "extractor", "model", "max_candidates", "workers", "output",
        "input", "overlap", "max_chunks", "timeout",
    ]

    def _row(fragments, idx: int, label: str, value: str) -> None:
        is_cursor = state["cursor"] == idx
        prefix = "  ▸ " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        fragments.append((style, f"{prefix}{label:<22}"))
        fragments.append(("class:value" if not is_cursor else style, value))
        fragments.append(("", "\n"))

    def _get_text():
        fragments = list(header_text("ppmlx memory config"))
        fragments.append(("class:dim", f"  {cfg_path}\n\n"))
        _row(fragments, 0, "Memory", f"◀ {'Enabled' if state['enabled'] else 'Disabled'} ▶")
        _row(fragments, 1, "Mode", f"◀ {modes[state['mode_index']]} ▶")
        _row(fragments, 2, "Extractor", f"◀ {extractors[state['extractor_index']]} (rule + model) ▶")
        is_cursor = state["cursor"] == 3
        prefix = "  ▸ " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        fragments.append((style, f"{prefix}Extraction Model      "))
        if state["editing_model"]:
            fragments.append(("class:value", state["model_buf"]))
            fragments.append(("class:value", "█"))
        else:
            fragments.append(("class:value" if not is_cursor else style, state["model"]))
        fragments.append(("", "\n"))
        _row(fragments, 4, "Max Candidates/Event", f"◀ {candidate_options[state['candidates_index']]} ▶")
        _row(fragments, 5, "Extractor Workers", f"◀ {worker_options[state['workers_index']]} ▶")
        _row(fragments, 6, "Output Limit", f"◀ {output_options[state['output_index']]} tokens ▶")
        _row(fragments, 7, "Input Limit", f"◀ {input_options[state['input_index']]} est. tokens ▶")
        _row(fragments, 8, "Chunk Overlap", f"◀ {overlap_options[state['overlap_index']]} est. tokens ▶")
        _row(fragments, 9, "Max Chunks/Event", f"◀ {chunk_options[state['chunks_index']]} ▶")
        _row(fragments, 10, "Extraction Timeout", f"◀ {timeout_options[state['timeout_index']]:g}s ▶")
        fragments.append(("", "\n"))
        if state["saved_flash"]:
            fragments.append(("class:checked", "                         ✓ saved\n"))
        elif state["dirty"]:
            fragments.append(("class:unsaved", "                         • unsaved changes\n"))
        else:
            fragments.append(("", "\n"))
        fragments.append(("", "\n"))
        if state["editing_model"]:
            fragments.append(("class:footer", "type model name • enter confirm • esc cancel"))
        else:
            fragments.append(("class:footer", "↑↓ navigate • ←→ cycle • enter edit model • s save • esc quit"))
        return fragments

    def _save() -> None:
        import tomli_w

        data.setdefault("memory", {})["enabled"] = state["enabled"]
        data.setdefault("memory", {})["mode"] = modes[state["mode_index"]]
        data.setdefault("memory", {})["extractor"] = extractors[state["extractor_index"]]
        data.setdefault("memory", {})["extraction_model"] = state["model"]
        data.setdefault("memory", {})["max_candidates_per_event"] = candidate_options[state["candidates_index"]]
        data.setdefault("memory", {})["extraction_workers"] = worker_options[state["workers_index"]]
        data.setdefault("memory", {})["extraction_max_tokens"] = output_options[state["output_index"]]
        data.setdefault("memory", {})["extraction_input_tokens"] = input_options[state["input_index"]]
        data.setdefault("memory", {})["extraction_overlap_tokens"] = overlap_options[state["overlap_index"]]
        data.setdefault("memory", {})["extraction_max_chunks_per_event"] = chunk_options[state["chunks_index"]]
        data.setdefault("memory", {})["extraction_timeout_seconds"] = timeout_options[state["timeout_index"]]
        with open(cfg_path, "wb") as f:
            tomli_w.dump(data, f)
        state["dirty"] = False
        state["saved_flash"] = True

    def _cycle(index_key: str, options: list, delta: int) -> None:
        state[index_key] = (state[index_key] + delta) % len(options)
        state["dirty"] = True

    def _cycle_current(delta: int) -> None:
        cursor = state["cursor"]
        if cursor == 0:
            state["enabled"] = not state["enabled"]
            if not state["enabled"]:
                state["mode_index"] = modes.index("off")
            elif modes[state["mode_index"]] == "off":
                state["mode_index"] = modes.index("shadow")
            state["dirty"] = True
        elif cursor == 1:
            _cycle("mode_index", modes, delta)
            state["enabled"] = modes[state["mode_index"]] != "off"
        elif cursor == 2:
            _cycle("extractor_index", extractors, delta)
        elif cursor == 4:
            _cycle("candidates_index", candidate_options, delta)
        elif cursor == 5:
            _cycle("workers_index", worker_options, delta)
        elif cursor == 6:
            _cycle("output_index", output_options, delta)
        elif cursor == 7:
            _cycle("input_index", input_options, delta)
        elif cursor == 8:
            _cycle("overlap_index", overlap_options, delta)
        elif cursor == 9:
            _cycle("chunks_index", chunk_options, delta)
        elif cursor == 10:
            _cycle("timeout_index", timeout_options, delta)

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        if state["editing_model"]:
            return
        if state["cursor"] > 0:
            state["cursor"] -= 1
            state["saved_flash"] = False

    @kb.add("down")
    def _down(event):
        if state["editing_model"]:
            return
        if state["cursor"] < len(items) - 1:
            state["cursor"] += 1
            state["saved_flash"] = False

    @kb.add("left")
    def _left(event):
        if not state["editing_model"]:
            state["saved_flash"] = False
            _cycle_current(-1)

    @kb.add("right")
    def _right(event):
        if not state["editing_model"]:
            state["saved_flash"] = False
            _cycle_current(1)

    @kb.add("enter")
    def _enter(event):
        state["saved_flash"] = False
        if state["editing_model"]:
            state["model"] = state["model_buf"] or "gemma-4-e2b"
            state["editing_model"] = False
            state["dirty"] = True
        elif state["cursor"] == 3:
            state["editing_model"] = True
            state["model_buf"] = state["model"]

    @kb.add("escape")
    def _escape(event):
        if state["editing_model"]:
            state["editing_model"] = False
            return
        if state["dirty"]:
            _save()
        event.app.exit(result=None)

    @kb.add("backspace")
    def _backspace(event):
        if state["editing_model"]:
            state["model_buf"] = state["model_buf"][:-1]

    @kb.add("s")
    def _save_key(event):
        if state["editing_model"]:
            state["model_buf"] += "s"
            return
        _save()

    @kb.add("<any>")
    def _char(event):
        ch = event.data
        if state["editing_model"] and ch.isprintable() and len(ch) == 1:
            state["model_buf"] += ch

    app = Application(
        layout=Layout(Window(content=FormattedTextControl(_get_text), always_hide_cursor=True)),
        key_bindings=kb,
        style=get_style(),
        full_screen=True,
        mouse_support=False,
    )
    app.run()


def _normalize_extractor(value: object) -> str:
    raw = str(value).strip().lower().replace("-", "_")
    if raw in {
        "hybrid", "rule", "rules", "rule_based", "regex",
        "model_memory_json", "memory_model_json", "model_json_memory", "strict_json_memory",
        "llm", "llm_json", "json_llm", "gemma_json",
    }:
        return "hybrid"
    return "hybrid"


def _option_index(options: list, value: object) -> int:
    try:
        typed_value = type(options[0])(value)
    except (TypeError, ValueError):
        typed_value = options[0]
    if typed_value not in options:
        options.append(typed_value)
        options.sort()
    return options.index(typed_value)

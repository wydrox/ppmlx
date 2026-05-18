"""Config editor TUI using prompt_toolkit."""
from __future__ import annotations


def config_menu() -> None:
    """Show an interactive config editor."""
    import tomllib

    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    from ppmlx.config import get_ppmlx_dir
    from ppmlx.tui._style import get_style, header_text

    cfg_path = get_ppmlx_dir() / "config.toml"

    # Load existing config
    data: dict = {}
    try:
        with open(cfg_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        pass

    hf_token = data.get("auth", {}).get("hf_token", "")
    ta_modes = ["off", "no_tools_only", "all"]
    refresh_modes = ["always", "weekly", "monthly", "never"]
    ta_current = data.get("tool_awareness", {}).get("mode", "no_tools_only")
    if ta_current not in ta_modes:
        ta_current = "no_tools_only"
    registry_limit_options = [10, 25, 50, 75, 100]
    refresh_current = data.get("registry", {}).get("refresh", "weekly")
    if refresh_current not in refresh_modes:
        refresh_current = "weekly"
    try:
        registry_limit_current = int(data.get("registry", {}).get("display_limit", 50))
    except (TypeError, ValueError):
        registry_limit_current = 50
    registry_limit_current = max(1, min(100, registry_limit_current))
    if registry_limit_current not in registry_limit_options:
        registry_limit_options.append(registry_limit_current)
        registry_limit_options = sorted(set(registry_limit_options))
    analytics_enabled = data.get("analytics", {}).get("enabled", False)
    thinking_enabled = data.get("thinking", {}).get("enabled", True)
    reasoning_budget = data.get("thinking", {}).get("default_reasoning_budget", 2048)
    budget_options = [0, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    if reasoning_budget not in budget_options:
        budget_options.append(reasoning_budget)
        budget_options.sort()
    effort_base = data.get("thinking", {}).get("effort_base", 256)
    effort_base_options = [64, 128, 256, 512, 1024]
    if effort_base not in effort_base_options:
        effort_base_options.append(effort_base)
        effort_base_options.sort()
    max_tools_tokens = data.get("server", {}).get("max_tools_tokens", 6000)
    tools_options = [0, 3000, 6000, 12000, 24000]
    if max_tools_tokens not in tools_options:
        tools_options.append(max_tools_tokens)
        tools_options.sort()

    ta_labels = {"off": "Off", "no_tools_only": "No Tools Only", "all": "All"}
    refresh_labels = {"always": "Always", "weekly": "Weekly", "monthly": "Monthly", "never": "Never"}
    analytics_labels = {True: "Enabled", False: "Disabled"}
    thinking_labels = {True: "Enabled", False: "Disabled"}
    state = {
        "cursor": 0,
        "hf_token": hf_token,
        "ta_index": ta_modes.index(ta_current),
        "refresh_index": refresh_modes.index(refresh_current),
        "registry_limit_index": registry_limit_options.index(registry_limit_current),
        "analytics": analytics_enabled,
        "thinking": thinking_enabled,
        "budget_index": budget_options.index(reasoning_budget),
        "effort_base_index": effort_base_options.index(effort_base),
        "tools_index": tools_options.index(max_tools_tokens),
        "dirty": False,
        "editing_field": None,
        "edit_buf": "",
        "saved_flash": False,
    }

    items = ["hf_token", "tool_awareness", "thinking", "reasoning_budget", "effort_base", "max_tools_tokens", "registry_refresh", "registry_limit", "analytics"]

    def _mask_token(token: str) -> str:
        if not token:
            return "(not set)"
        if len(token) <= 4:
            return "\u2022" * len(token)
        return "\u2022" * (len(token) - 4) + token[-4:]

    def _get_text():
        fragments = list(header_text("ppmlx config"))

        # Config path
        fragments.append(("class:dim", f"  {cfg_path}\n\n"))

        # HF Token row
        is_cursor = state["cursor"] == 0
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        if state["editing_field"] == "hf_token":
            fragments.append((style, f"{prefix}HuggingFace Token    "))
            fragments.append(("class:value", state["edit_buf"]))
            fragments.append(("class:value", "\u2588"))
            fragments.append(("", "\n"))
        else:
            masked = _mask_token(state["hf_token"])
            fragments.append((style, f"{prefix}HuggingFace Token    "))
            fragments.append(("class:dim" if not is_cursor else style, masked))
            fragments.append(("", "\n"))

        # Tool Awareness row
        is_cursor = state["cursor"] == 1
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        ta_label = ta_labels[ta_modes[state["ta_index"]]]
        fragments.append((style, f"{prefix}Tool Awareness       "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {ta_label} \u25b6"))
        fragments.append(("", "\n"))

        # Thinking row
        is_cursor = state["cursor"] == 2
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        th_label = thinking_labels[state["thinking"]]
        fragments.append((style, f"{prefix}Thinking             "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {th_label} \u25b6"))
        fragments.append(("", "\n"))

        # Reasoning Budget row
        is_cursor = state["cursor"] == 3
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        budget_val = budget_options[state["budget_index"]]
        budget_label = "Off" if budget_val == 0 else f"{budget_val} tokens"
        fragments.append((style, f"{prefix}Reasoning Budget     "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {budget_label} \u25b6"))
        fragments.append(("", "\n"))

        # Effort Base row
        is_cursor = state["cursor"] == 4
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        eb_val = effort_base_options[state["effort_base_index"]]
        eb_label = f"{eb_val} (low={eb_val}, med={eb_val*4}, high={eb_val*32})"
        fragments.append((style, f"{prefix}Effort Base          "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {eb_label} \u25b6"))
        fragments.append(("", "\n"))

        # Max Tools Tokens row
        is_cursor = state["cursor"] == 5
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        tt_val = tools_options[state["tools_index"]]
        tt_label = "Unlimited" if tt_val == 0 else f"{tt_val} tokens"
        fragments.append((style, f"{prefix}Max Tools Tokens     "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {tt_label} \u25b6"))
        fragments.append(("", "\n"))

        # Registry auto-refresh row
        is_cursor = state["cursor"] == 6
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        refresh_label = refresh_labels[refresh_modes[state["refresh_index"]]]
        fragments.append((style, f"{prefix}Registry Refresh     "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {refresh_label} \u25b6"))
        fragments.append(("", "\n"))

        # Registry display limit row
        is_cursor = state["cursor"] == 7
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        limit_label = f"{registry_limit_options[state['registry_limit_index']]} models"
        fragments.append((style, f"{prefix}Pull Model Count     "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {limit_label} \u25b6"))
        fragments.append(("", "\n"))

        # Analytics row
        is_cursor = state["cursor"] == 8
        prefix = "  \u25b8 " if is_cursor else "    "
        style = "class:cursor" if is_cursor else ""
        an_label = analytics_labels[state["analytics"]]
        fragments.append((style, f"{prefix}Usage Analytics       "))
        fragments.append(("class:value" if not is_cursor else style, f"\u25c0 {an_label} \u25b6"))
        fragments.append(("", "\n"))

        fragments.append(("", "\n"))

        if state["saved_flash"]:
            fragments.append(("class:checked", "                         \u2713 saved\n"))
        elif state["dirty"]:
            fragments.append(("class:unsaved", "                         \u2022 unsaved changes\n"))
        else:
            fragments.append(("", "\n"))

        fragments.append(("", "\n"))
        if state["editing_field"]:
            fragments.append(("class:footer", "type value \u2022 enter confirm \u2022 esc cancel"))
        else:
            fragments.append(("class:footer", "\u2191\u2193 navigate \u2022 \u2190\u2192 cycle \u2022 enter edit text \u2022 s save \u2022 esc quit"))
        return fragments

    def _save():
        import tomli_w

        data.setdefault("auth", {})["hf_token"] = state["hf_token"]
        data.setdefault("tool_awareness", {})["mode"] = ta_modes[state["ta_index"]]
        data.setdefault("thinking", {})["enabled"] = state["thinking"]
        data.setdefault("thinking", {})["default_reasoning_budget"] = budget_options[state["budget_index"]]
        data.setdefault("thinking", {})["effort_base"] = effort_base_options[state["effort_base_index"]]
        data.setdefault("server", {})["max_tools_tokens"] = tools_options[state["tools_index"]]
        data.setdefault("registry", {})["refresh"] = refresh_modes[state["refresh_index"]]
        data.setdefault("registry", {})["display_limit"] = registry_limit_options[state["registry_limit_index"]]
        data.setdefault("analytics", {})["enabled"] = state["analytics"]
        with open(cfg_path, "wb") as f:
            tomli_w.dump(data, f)
        state["dirty"] = False
        state["saved_flash"] = True

    kb = KeyBindings()

    def _cycle(index_key: str, options: list, delta: int) -> None:
        state[index_key] = (state[index_key] + delta) % len(options)
        state["dirty"] = True

    def _cycle_current(delta: int) -> None:
        cursor = state["cursor"]
        if cursor == 1:
            _cycle("ta_index", ta_modes, delta)
        elif cursor == 2:
            state["thinking"] = not state["thinking"]
            state["dirty"] = True
        elif cursor == 3:
            _cycle("budget_index", budget_options, delta)
        elif cursor == 4:
            _cycle("effort_base_index", effort_base_options, delta)
        elif cursor == 5:
            _cycle("tools_index", tools_options, delta)
        elif cursor == 6:
            _cycle("refresh_index", refresh_modes, delta)
        elif cursor == 7:
            _cycle("registry_limit_index", registry_limit_options, delta)
        elif cursor == 8:
            state["analytics"] = not state["analytics"]
            state["dirty"] = True

    @kb.add("up")
    def _up(event):
        if state["editing_field"]:
            return
        if state["cursor"] > 0:
            state["cursor"] -= 1
            state["saved_flash"] = False

    @kb.add("down")
    def _down(event):
        if state["editing_field"]:
            return
        if state["cursor"] < len(items) - 1:
            state["cursor"] += 1
            state["saved_flash"] = False

    @kb.add("left")
    def _left(event):
        if state["editing_field"]:
            return
        state["saved_flash"] = False
        _cycle_current(-1)

    @kb.add("right")
    def _right(event):
        if state["editing_field"]:
            return
        state["saved_flash"] = False
        _cycle_current(1)

    @kb.add("enter")
    def _enter(event):
        state["saved_flash"] = False
        if state["editing_field"] == "hf_token":
            state["hf_token"] = state["edit_buf"]
            state["editing_field"] = None
            state["dirty"] = True
        elif state["cursor"] == 0:
            state["editing_field"] = "hf_token"
            state["edit_buf"] = state["hf_token"]
    @kb.add("escape")
    def _escape(event):
        if state["editing_field"]:
            state["editing_field"] = None
            return
        if state["dirty"]:
            _save()
        event.app.exit(result=None)

    @kb.add("backspace")
    def _backspace(event):
        if state["editing_field"]:
            state["edit_buf"] = state["edit_buf"][:-1]

    @kb.add("s")
    def _save_key(event):
        if state["editing_field"]:
            state["edit_buf"] += "s"
            return
        _save()

    @kb.add("<any>")
    def _char(event):
        ch = event.data
        if state["editing_field"] and ch.isprintable() and len(ch) == 1:
            state["edit_buf"] += ch

    body = Window(
        content=FormattedTextControl(_get_text),
        always_hide_cursor=True,
    )

    app = Application(
        layout=Layout(body),
        key_bindings=kb,
        style=get_style(),
        full_screen=True,
        mouse_support=False,
    )

    app.run()

"""Launch action menu using prompt_toolkit."""
from __future__ import annotations

import shutil


def launch_menu(
    *,
    preset_action: str | None = None,
    command_str: str = "ppmlx launch",
) -> tuple[str | None, str | None]:
    """Show launch menu. Returns (action_key, model_alias) or (None, None)."""
    from ppmlx.tui._model_picker import pick_model

    # If preset_action given, skip action menu and go straight to model picker
    if preset_action is not None:
        model = pick_model(command_str=command_str)
        if model is None:
            return (None, None)
        return (preset_action, model)

    action = _show_action_menu(command_str=command_str)
    if action is None:
        return (None, None)

    model = pick_model(command_str=f"{command_str} > {action}")
    if model is None:
        return (None, None)
    return (action, model)


def _show_action_menu(*, command_str: str = "ppmlx launch") -> str | None:
    """Display the action picker and return the selected action key or None."""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    from ppmlx.cli import _LAUNCH_ITEMS
    from ppmlx.tui._style import get_style, header_text

    items = _LAUNCH_ITEMS

    # Check which items are installed
    installed = {}
    for item in items:
        if item.cmd == "":
            installed[item.key] = True
        else:
            installed[item.key] = shutil.which(item.cmd) is not None

    selectable = [i for i, item in enumerate(items) if installed[item.key]]
    state = {"cursor": selectable[0] if selectable else 0}

    def _get_text():
        fragments = list(header_text(command_str))

        for i, item in enumerate(items):
            is_cursor = i == state["cursor"]
            is_installed = installed[item.key]

            prefix = "  \u25b8 " if is_cursor else "    "

            if not is_installed:
                style = "class:disabled"
                suffix = "  (not installed)"
            elif is_cursor:
                style = "class:cursor"
                suffix = ""
            else:
                style = ""
                suffix = ""

            fragments.append((style, f"{prefix}{item.label}"))
            if suffix:
                fragments.append(("class:disabled", suffix))
            fragments.append(("", "\n"))

            # Description line
            desc_prefix = "    "
            if is_cursor:
                fragments.append(("class:desc", f"{desc_prefix}{item.desc}\n"))
            else:
                fragments.append(("class:desc", f"{desc_prefix}{item.desc}\n"))

            fragments.append(("", "\n"))

        fragments.append(("class:footer", "\u2191\u2193 navigate \u2022 enter launch \u2022 esc quit"))
        return fragments

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        try:
            pos = selectable.index(state["cursor"])
            if pos > 0:
                state["cursor"] = selectable[pos - 1]
        except ValueError:
            if selectable:
                state["cursor"] = selectable[0]

    @kb.add("down")
    def _down(event):
        try:
            pos = selectable.index(state["cursor"])
            if pos < len(selectable) - 1:
                state["cursor"] = selectable[pos + 1]
        except ValueError:
            if selectable:
                state["cursor"] = selectable[0]

    @kb.add("enter")
    def _enter(event):
        idx = state["cursor"]
        if idx < len(items) and installed[items[idx].key]:
            event.app.exit(result=items[idx].key)

    @kb.add("escape")
    def _escape(event):
        event.app.exit(result=None)

    body = Window(
        content=FormattedTextControl(_get_text),
        always_hide_cursor=True,
    )

    app: Application[str | None] = Application(
        layout=Layout(body),
        key_bindings=kb,
        style=get_style(),
        full_screen=True,
        mouse_support=False,
    )

    return app.run()

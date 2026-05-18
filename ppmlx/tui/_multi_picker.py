"""Multi-select model picker using prompt_toolkit."""
from __future__ import annotations


def pick_models(*, local_only: bool = False) -> list[str]:
    """Show a multi-select model picker. Returns list of selected aliases."""
    from prompt_toolkit import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, HSplit, Window, ScrollOffsets
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.data_structures import Point

    from ppmlx.cli import _build_picker_rows, _visible_rows
    from ppmlx.tui._style import (
        get_style, header_text,
        render_model_row, render_table_header, render_section_title,
    )

    all_rows = _build_picker_rows(local_only=local_only)
    try:
        from ppmlx.registry_fetch import cache_status_text
        registry_status = cache_status_text()
    except Exception:
        registry_status = "last refresh: unknown"

    state: dict = {"cursor": 0, "search": "", "selected": set()}

    def _selectable_indices(rows):
        return [i for i, r in enumerate(rows) if r.section_header is None]

    def _filtered():
        return _visible_rows(all_rows, state["search"])

    def _clamp_cursor(rows):
        indices = _selectable_indices(rows)
        if not indices:
            state["cursor"] = 0
            return
        if state["cursor"] not in indices:
            state["cursor"] = indices[0]

    def _get_header():
        fragments = list(header_text("ppmlx"))
        fragments.append(("", "Search: "))
        fragments.append(("class:value", state["search"]))
        fragments.append(("class:value", "\u2588"))
        if not local_only:
            fragments.append(("class:dim", f"  Registry {registry_status}"))
        fragments.append(("", "\n"))
        return fragments

    def _cursor_line():
        rows = _filtered()
        line = 0
        for i, row in enumerate(rows):
            if row.section_header is not None:
                line += 4  # blank + title + header + separator
                continue
            if i == state["cursor"]:
                return line
            line += 1
        return 0

    def _get_list():
        rows = _filtered()
        _clamp_cursor(rows)
        fragments = []

        if not rows:
            fragments.append(("class:dim", "  No models found.\n"))
            return fragments

        for i, row in enumerate(rows):
            if row.section_header is not None:
                fragments.extend(render_section_title(row.section_header))
                fragments.extend(render_table_header(show_checkbox=True))
                continue
            checked = row.alias in state["selected"]
            checkbox = "[x]" if checked else "[ ]"
            fragments.extend(render_model_row(
                row, is_cursor=i == state["cursor"], checkbox=checkbox,
            ))

        return fragments

    def _get_footer():
        n = len(state["selected"])
        parts = []
        if n:
            parts.append(("class:checked", f"  {n} selected  "))
        parts.append(("class:footer", "\u2191\u2193 navigate \u2022 space toggle \u2022 enter confirm \u2022 esc cancel \u2022 type to search"))
        return parts

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        rows = _filtered()
        indices = _selectable_indices(rows)
        if not indices:
            return
        try:
            pos = indices.index(state["cursor"])
            if pos > 0:
                state["cursor"] = indices[pos - 1]
        except ValueError:
            state["cursor"] = indices[0]

    @kb.add("down")
    def _down(event):
        rows = _filtered()
        indices = _selectable_indices(rows)
        if not indices:
            return
        try:
            pos = indices.index(state["cursor"])
            if pos < len(indices) - 1:
                state["cursor"] = indices[pos + 1]
        except ValueError:
            state["cursor"] = indices[0]

    @kb.add(" ")
    def _space(event):
        rows = _filtered()
        indices = _selectable_indices(rows)
        if state["cursor"] in indices:
            row = rows[state["cursor"]]
            alias = row.alias
            if alias in state["selected"]:
                state["selected"].discard(alias)
            else:
                state["selected"].add(alias)

    @kb.add("enter")
    def _enter(event):
        event.app.exit(result=sorted(state["selected"]))

    @kb.add("escape")
    def _escape(event):
        event.app.exit(result=[])

    @kb.add("backspace")
    def _backspace(event):
        if state["search"]:
            state["search"] = state["search"][:-1]
            state["cursor"] = 0

    @kb.add("<any>")
    def _char(event):
        ch = event.data
        if ch == " ":
            return
        if ch.isprintable() and len(ch) == 1:
            state["search"] += ch
            state["cursor"] = 0

    header_window = Window(
        content=FormattedTextControl(_get_header),
        height=4,
        always_hide_cursor=True,
    )

    list_control = FormattedTextControl(_get_list)
    list_control.get_cursor_position = lambda: Point(x=0, y=_cursor_line())

    list_window = Window(
        content=list_control,
        always_hide_cursor=True,
        scroll_offsets=ScrollOffsets(top=2, bottom=2),
    )

    footer_window = Window(
        content=FormattedTextControl(_get_footer),
        height=1,
        always_hide_cursor=True,
    )

    app = Application(
        layout=Layout(HSplit([header_window, list_window, footer_window])),
        key_bindings=kb,
        style=get_style(),
        full_screen=True,
        mouse_support=False,
    )

    return app.run()

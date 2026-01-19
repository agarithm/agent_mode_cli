from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional


def _in_container() -> bool:
    return os.getenv("AI_IN_CONTAINER", "").lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class ConfirmRequest:
    tool_name: str
    arguments: Mapping[str, Any]


class TextualPromptBridge:
    """Thread-safe confirmation bridge for Textual.

    The agent loop runs in a worker thread under the Textual UI (via asyncio.to_thread).
    This bridge lets that thread synchronously ask the user for permission by
    scheduling a modal screen on the Textual app thread.
    """

    def __init__(self) -> None:
        self._app: Any = None
        self._lock = threading.Lock()

    def attach_app(self, app: Any) -> None:
        with self._lock:
            self._app = app

    def confirm(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        approve_all: bool,
        debug: bool = False,
    ) -> str:
        """Return one of: 'yes', 'no', 'all'."""

        # Keep parity with CLI behavior.
        if _in_container():
            return "yes"
        if approve_all:
            return "yes"

        with self._lock:
            app = self._app

        if app is None:
            # No UI available; safest default is deny.
            if debug:
                try:
                    import sys

                    print("[debug] TextualConfirmBridge: no app attached; defaulting to no", file=sys.stderr)
                except Exception:
                    pass
            return "no"

        try:
            from textual.containers import Horizontal, Vertical
            from textual.screen import ModalScreen
            from textual.widgets import Button, Static
        except Exception:
            # If Textual isn't importable for any reason, default to deny.
            return "no"

        title = f"Confirm: {tool_name}"
        try:
            pretty_args = json.dumps(dict(arguments), indent=2)
        except Exception:
            pretty_args = str(arguments)
        body = (
            "This tool may modify files or system state.\n\n"
            "Arguments:\n"
            f"{pretty_args}"
        )

        done = threading.Event()
        result: dict[str, str] = {"value": "no"}

        class ConfirmScreen(ModalScreen[str]):
            can_focus = True
            AUTO_FOCUS = "#no"
            DEFAULT_CSS = """
            ConfirmScreen {
                align: center middle;
            }
            #dialog {
                width: 80%;
                max-width: 100;
                min-width: 50;
                border: round $primary;
                background: $panel;
                padding: 1 2;
            }
            #title {
                text-style: bold;
                margin-bottom: 1;
            }
            #body {
                height: auto;
                max-height: 20;
                overflow-y: auto;
                margin-bottom: 1;
            }
            #buttons {
                height: auto;
                align-horizontal: right;
            }
            """

            try:
                from textual.binding import Binding

                BINDINGS = [
                    Binding("y", "yes", "Yes", priority=True),
                    Binding("n", "no", "No", priority=True),
                    Binding("a", "all", "All", priority=True),
                    Binding("escape", "no", "Cancel", priority=True),
                ]
            except Exception:  # pragma: no cover
                BINDINGS = [
                    ("y", "yes", "Yes"),
                    ("n", "no", "No"),
                    ("a", "all", "All"),
                    ("escape", "no", "Cancel"),
                ]

            def compose(self):  # type: ignore[override]
                with Vertical(id="dialog"):
                    yield Static(title, id="title", markup=False)
                    yield Static(body, id="body", markup=False)
                    with Horizontal(id="buttons"):
                        yield Button("No", id="no", variant="error")
                        yield Button("Yes", id="yes", variant="success")
                        yield Button("All", id="all", variant="primary")

            def on_mount(self) -> None:  # type: ignore[override]
                # Ensure modal receives keyboard input (otherwise underlying Input may keep focus).
                def _focus() -> None:
                    try:
                        self.app.set_focus(self.query_one("#no", Button), scroll_visible=False)
                    except Exception:
                        pass
                try:
                    self.call_after_refresh(_focus)
                except Exception:
                    try:
                        self.call_later(_focus)
                    except Exception:
                        _focus()

            # Note: we rely on priority key bindings; on_key is intentionally omitted
            # because focused widgets may swallow Key events before they bubble.

            def action_yes(self) -> None:
                self.dismiss("yes")

            def action_no(self) -> None:
                self.dismiss("no")

            def action_all(self) -> None:
                self.dismiss("all")

            def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
                bid = (event.button.id or "").lower()
                if bid == "yes":
                    self.dismiss("yes")
                elif bid == "all":
                    self.dismiss("all")
                else:
                    self.dismiss("no")

        def _on_result(value: Optional[str]) -> None:
            result["value"] = (value or "no")
            done.set()

        def _push() -> None:
            try:
                app.push_screen(ConfirmScreen(), _on_result)
            except TypeError:
                # Some versions use keyword-only callback.
                try:
                    app.push_screen(ConfirmScreen(), callback=_on_result)
                except Exception:
                    _on_result("no")
            except Exception:
                _on_result("no")

        # Schedule modal on the UI thread.
        try:
            if hasattr(app, "call_from_thread"):
                app.call_from_thread(_push)
            else:
                # Best-effort fallback.
                app.call_later(_push)
        except Exception:
            return "no"

        done.wait()
        return result["value"]

    def select_one(
        self,
        *,
        title: str,
        options: list[str] | tuple[str, ...] | Any,
        default: Optional[str] = None,
        allow_cancel: bool = True,
    ) -> Optional[str]:
        with self._lock:
            app = self._app
        if app is None:
            return None

        items = [str(o) for o in (options or []) if str(o).strip()]
        if not items:
            return None

        try:
            from textual.containers import Horizontal, Vertical
            from textual.screen import ModalScreen
            from textual.widgets import Button, ListItem, ListView, Static
        except Exception:
            return None

        done = threading.Event()
        result: dict[str, Optional[str]] = {"value": None}

        class ChoiceScreen(ModalScreen[Optional[str]]):
            can_focus = True
            AUTO_FOCUS = "#list"
            DEFAULT_CSS = """
            ChoiceScreen {
                align: center middle;
            }
            #dialog {
                width: 80%;
                max-width: 100;
                min-width: 50;
                border: round $primary;
                background: $panel;
                padding: 1 2;
            }
            #title { text-style: bold; margin-bottom: 1; }
            #list { height: 12; overflow-y: auto; margin-bottom: 1; }
            #buttons { height: auto; align-horizontal: right; }
            """

            BINDINGS = [
                ("escape", "cancel", "Cancel"),
                ("enter", "accept", "Accept"),
            ]

            def compose(self):  # type: ignore[override]
                with Vertical(id="dialog"):
                    yield Static(title, id="title", markup=False)
                    yield ListView(*[ListItem(Static(label, markup=False)) for label in items], id="list")
                    with Horizontal(id="buttons"):
                        if allow_cancel:
                            yield Button("Cancel", id="cancel", variant="error")
                        yield Button("OK", id="ok", variant="primary")

            def on_mount(self) -> None:  # type: ignore[override]
                def _focus() -> None:
                    try:
                        lv = self.query_one("#list", ListView)
                        if default and default in items:
                            idx = items.index(default)
                            lv.index = idx
                        self.app.set_focus(lv, scroll_visible=False)
                    except Exception:
                        pass
                try:
                    self.call_after_refresh(_focus)
                except Exception:
                    try:
                        self.call_later(_focus)
                    except Exception:
                        _focus()

            def action_cancel(self) -> None:
                self.dismiss(None)

            def action_accept(self) -> None:
                try:
                    lv = self.query_one("#list", ListView)
                    idx = int(getattr(lv, "index", 0) or 0)
                    if 0 <= idx < len(items):
                        self.dismiss(items[idx])
                        return
                except Exception:
                    pass
                self.dismiss(None)

            def on_list_view_selected(self, event: ListView.Selected) -> None:  # type: ignore[override]
                # Double-click / enter on an item.
                try:
                    idx = int(getattr(event, "index", 0) or 0)
                    if 0 <= idx < len(items):
                        self.dismiss(items[idx])
                        return
                except Exception:
                    pass
                self.dismiss(None)

            def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
                bid = (event.button.id or "").lower()
                if bid == "ok":
                    self.action_accept()
                else:
                    self.action_cancel()

        def _on_result(value: Optional[str]) -> None:
            result["value"] = value
            done.set()

        def _push() -> None:
            try:
                app.push_screen(ChoiceScreen(), _on_result)
            except TypeError:
                try:
                    app.push_screen(ChoiceScreen(), callback=_on_result)
                except Exception:
                    _on_result(None)
            except Exception:
                _on_result(None)

        try:
            if hasattr(app, "call_from_thread"):
                app.call_from_thread(_push)
            else:
                app.call_later(_push)
        except Exception:
            return None

        done.wait()
        return result["value"]

    def prompt_text(
        self,
        *,
        title: str,
        placeholder: str = "",
        default: str = "",
        allow_cancel: bool = True,
    ) -> Optional[str]:
        with self._lock:
            app = self._app
        if app is None:
            return None

        try:
            from textual.containers import Horizontal, Vertical
            from textual.screen import ModalScreen
            from textual.widgets import Button, Input, Static
        except Exception:
            return None

        done = threading.Event()
        result: dict[str, Optional[str]] = {"value": None}

        class TextScreen(ModalScreen[Optional[str]]):
            can_focus = True
            AUTO_FOCUS = "#value"
            DEFAULT_CSS = """
            TextScreen { align: center middle; }
            #dialog {
                width: 80%;
                max-width: 100;
                min-width: 50;
                border: round $primary;
                background: $panel;
                padding: 1 2;
            }
            #title { text-style: bold; margin-bottom: 1; }
            #buttons { height: auto; align-horizontal: right; margin-top: 1; }
            """

            BINDINGS = [
                ("escape", "cancel", "Cancel"),
                ("enter", "ok", "OK"),
            ]

            def compose(self):  # type: ignore[override]
                with Vertical(id="dialog"):
                    yield Static(title, id="title", markup=False)
                    yield Input(value=default or "", placeholder=placeholder or "", id="value")
                    with Horizontal(id="buttons"):
                        if allow_cancel:
                            yield Button("Cancel", id="cancel", variant="error")
                        yield Button("OK", id="ok", variant="primary")

            def on_mount(self) -> None:  # type: ignore[override]
                def _focus() -> None:
                    try:
                        self.app.set_focus(self.query_one("#value", Input), scroll_visible=False)
                    except Exception:
                        pass
                try:
                    self.call_after_refresh(_focus)
                except Exception:
                    try:
                        self.call_later(_focus)
                    except Exception:
                        _focus()

            def action_cancel(self) -> None:
                self.dismiss(None)

            def action_ok(self) -> None:
                try:
                    value = self.query_one("#value", Input).value
                except Exception:
                    value = None
                if value is None:
                    self.dismiss(None)
                else:
                    self.dismiss(str(value))

            def on_button_pressed(self, event: Button.Pressed) -> None:  # type: ignore[override]
                bid = (event.button.id or "").lower()
                if bid == "ok":
                    self.action_ok()
                else:
                    self.action_cancel()

        def _on_result(value: Optional[str]) -> None:
            result["value"] = value
            done.set()

        def _push() -> None:
            try:
                app.push_screen(TextScreen(), _on_result)
            except TypeError:
                try:
                    app.push_screen(TextScreen(), callback=_on_result)
                except Exception:
                    _on_result(None)
            except Exception:
                _on_result(None)

        try:
            if hasattr(app, "call_from_thread"):
                app.call_from_thread(_push)
            else:
                app.call_later(_push)
        except Exception:
            return None

        done.wait()
        return result["value"]


# Back-compat alias
TextualConfirmBridge = TextualPromptBridge

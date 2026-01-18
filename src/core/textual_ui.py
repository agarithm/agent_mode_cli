from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional

from core.repl import is_exit_command, renderables_for_result


@dataclass(frozen=True)
class TextualReplConfig:
    refresh_seconds: float = 0.5


def run_textual_repl(
    *,
    initial_line: Optional[str],
    before_first_prompt: Callable[[], None],
    process_line: Callable[[str], str],
    after_each_prompt: Callable[[], None],
    prompt_provider: Optional[Callable[[], str]] = None,
    context_version: Optional[Callable[[], int]] = None,
    context_snapshot: Optional[Callable[[], str]] = None,
    context_delta: Optional[Callable[[int], str]] = None,
    config: TextualReplConfig | None = None,
) -> int:
    """Run a full-screen Textual REPL.

    This UI is single-threaded (as terminals should be). Long-running `process_line`
    work is executed in a background thread so input and other panes remain responsive.
    """

    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.actions import SkipAction
        from textual.events import Key, MouseScrollDown, MouseScrollUp
        from textual.widgets import ContentSwitcher, Footer, Header, Input, Static, RichLog, TextArea
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("textual is not installed") from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"failed to import Textual UI components: {exc}") from exc

    cfg = config or TextualReplConfig()

    class AgentTui(App[int]):
        CSS = """
        #body { height: 100%; }
        #main { layout: horizontal; height: 1fr; }
        #left { width: 1fr; }
        #right { width: 48; min-width: 36; }

        #transcript_switcher { height: 1fr; }
        #prompt_line { height: 1; }
        #command { height: 3; }

        #context { height: 1fr; }
        """

        BINDINGS = [
            ("ctrl+q", "quit_clean", "Quit"),
            ("f2", "toggle_copy_mode", "Copy Mode"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._context_last_version: int = -1
            self._did_init_session: bool = False
            self._copy_mode: bool = False
            self._raw_transcript_lines: list[str] = []
            self._context_follow: bool = True

        def _update_context_follow(self) -> None:
            try:
                view = self.query_one("#context", TextArea)
                self._context_follow = bool(getattr(view, "is_vertical_scroll_end", True))
            except Exception:
                # Default to following if we can't determine scroll state.
                self._context_follow = True

        def _safe_scroll_context_end(self) -> None:
            try:
                view = self.query_one("#context", TextArea)
            except Exception:
                return

            try:
                # Avoid action_scroll_end (can raise SkipAction). Use Widget.scroll_end directly.
                view.scroll_end(animate=False, immediate=True, force=True, x_axis=False, y_axis=True)
            except SkipAction:
                return
            except Exception:
                return

        def on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
            try:
                if getattr(event.control, "id", None) == "context":
                    self._context_follow = False
            except Exception:
                return

        def on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
            try:
                if getattr(event.control, "id", None) != "context":
                    return
            except Exception:
                return

            # After the scroll is applied, recompute whether we're at the end.
            try:
                self.call_later(self._update_context_follow)
            except Exception:
                pass

        def on_key(self, event: Key) -> None:
            focused = getattr(self, "focused", None)
            if getattr(focused, "id", None) != "context":
                return

            key = (event.key or "").lower()
            if key in {"pageup", "home"}:
                self._context_follow = False
                return

            if key in {"pagedown", "end"}:
                try:
                    self.call_later(self._update_context_follow)
                except Exception:
                    pass

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="left"):
                    with ContentSwitcher(id="transcript_switcher", initial="transcript_rich"):
                        yield RichLog(id="transcript_rich", wrap=True)
                        yield TextArea(
                            "",
                            id="transcript_raw",
                            read_only=True,
                            soft_wrap=True,
                            show_cursor=False,
                            highlight_cursor_line=False,
                            show_line_numbers=False,
                        )
                    yield Static("", id="prompt_line")
                    yield Input(placeholder="> ", id="command")
                with Vertical(id="right"):
                    yield Static("Context", id="context_title")
                    yield TextArea(
                        "",
                        id="context",
                        read_only=True,
                        soft_wrap=True,
                        show_cursor=False,
                        highlight_cursor_line=False,
                        show_line_numbers=False,
                    )
            yield Footer()

        def action_toggle_copy_mode(self) -> None:
            self._copy_mode = not self._copy_mode
            switcher = self.query_one("#transcript_switcher", ContentSwitcher)
            if self._copy_mode:
                raw = self.query_one("#transcript_raw", TextArea)
                raw.text = "\n".join(self._raw_transcript_lines)
                switcher.current = "transcript_raw"
                raw.focus()
            else:
                switcher.current = "transcript_rich"
                self.query_one("#command", Input).focus()

        def action_quit_clean(self) -> None:
            # Exit without propagating a SystemExit so the caller can run
            # post-REPL cleanup (e.g., git auto-branch cleanup).
            try:
                transcript = self.query_one("#transcript_rich", RichLog)
                transcript.write("Exiting.")
                self._append_transcript_raw("Exiting.")
            except Exception:
                pass
            self.exit(0)

        def _append_transcript_raw(self, line: str) -> None:
            self._raw_transcript_lines.append(line)
            if self._copy_mode:
                raw = self.query_one("#transcript_raw", TextArea)
                raw.text = "\n".join(self._raw_transcript_lines)

        def on_mount(self) -> None:
            self.query_one("#command", Input).focus()

            # Enable text selection in panes when supported by this Textual version.
            for area_id in ("#context", "#transcript_raw"):
                try:
                    area = self.query_one(area_id, TextArea)
                except Exception:
                    continue
                # Ensure the user can scroll these panes.
                try:
                    area.allow_vertical_scroll = True
                except Exception:
                    pass
                # Some versions expose allow_select, some enable selection via actions only.
                for attr in ("allow_select", "ALLOW_SELECT"):
                    try:
                        if hasattr(area, attr):
                            setattr(area, attr, True)
                            break
                    except Exception:
                        continue

            # Ensure the system prompt (and other session-start context) is always
            # injected even before the user submits anything.
            if not self._did_init_session:
                try:
                    before_first_prompt()
                finally:
                    self._did_init_session = True

            self._refresh_context()
            self._update_prompt()
            # Start the context pane at the bottom (tail-follow) on launch.
            self._context_follow = True
            try:
                self.call_after_refresh(self._safe_scroll_context_end)
            except Exception:
                try:
                    self.call_later(self._safe_scroll_context_end)
                except Exception:
                    self._safe_scroll_context_end()
            self.set_interval(cfg.refresh_seconds, self._refresh_context)

        def _update_prompt(self) -> None:
            label = "> "
            if prompt_provider is not None:
                try:
                    label = prompt_provider() or "> "
                except Exception:
                    label = "> "
            self.query_one("#prompt_line", Static).update(label)

        def _refresh_context(self) -> None:
            if context_version is None or context_snapshot is None:
                return
            try:
                version = int(context_version())
            except Exception:
                return
            if version == self._context_last_version:
                return

            view = self.query_one("#context", TextArea)
            follow = bool(self._context_follow)

            # Prefer incremental updates (append-only) to avoid disorienting repaints.
            if context_delta is not None and self._context_last_version >= 0 and version > self._context_last_version:
                try:
                    delta = context_delta(self._context_last_version) or ""
                except Exception:
                    delta = ""
                if delta:
                    try:
                        view.insert(delta, location=view.document.end)
                    except Exception:
                        # Fallback: set the full snapshot if insertion fails.
                        try:
                            view.text = (view.text or "") + delta
                        except Exception:
                            pass
            else:
                try:
                    snapshot = context_snapshot() or ""
                except Exception:
                    snapshot = "(error rendering context)"
                view.text = snapshot

            self._context_last_version = version

            # Auto-scroll only if the user hasn't scrolled away (tail -f behavior).
            if follow:
                # Do an immediate best-effort scroll first to avoid visible bounce.
                self._safe_scroll_context_end()
                # Scroll after the widget has processed the document update;
                # immediate scroll can be overwritten by the TextArea refresh.
                try:
                    self.call_after_refresh(self._safe_scroll_context_end)
                except Exception:
                    try:
                        self.call_later(self._safe_scroll_context_end)
                    except Exception:
                        self._safe_scroll_context_end()

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            line = (event.value or "").strip()
            event.input.value = ""
            if not line:
                return

            transcript = self.query_one("#transcript_rich", RichLog)
            transcript.write(f"> {line}")
            self._append_transcript_raw(f"> {line}")

            if is_exit_command(line):
                transcript.write("Exiting.")
                self._append_transcript_raw("Exiting.")
                self.exit(0)
                return

            # Ensure the initial session prompt/context is added before first command.
            # We do this lazily so the UI can mount first.
            if not self._did_init_session:
                try:
                    before_first_prompt()
                finally:
                    self._did_init_session = True
                    self._refresh_context()
                    self._update_prompt()

            event.input.disabled = True
            try:
                result = await asyncio.to_thread(process_line, line)
            finally:
                try:
                    after_each_prompt()
                finally:
                    event.input.disabled = False
                    event.input.focus()
                    self._refresh_context()
                    self._update_prompt()

            (result or "")
            for part in renderables_for_result(result):
                transcript.write(part)

            self._append_transcript_raw(">>>>>>>>>")
            result_text = (result or "").rstrip()
            if result_text:
                for out_line in result_text.splitlines():
                    self._append_transcript_raw(out_line)
            self._append_transcript_raw("")

    # If an initial line exists, we run it by seeding the input after mount.
    # This is implemented as a best-effort QoL feature.
    app = AgentTui()

    if initial_line is not None and initial_line.strip():
        seed = initial_line.strip()

        async def _seed_after_ready() -> None:
            await asyncio.sleep(0)
            app._refresh_context()  # type: ignore[attr-defined]
            app._update_prompt()  # type: ignore[attr-defined]
            # Run the initial command as a background task.
            transcript = app.query_one("#transcript_rich", RichLog)
            transcript.write(f"> {seed}")

            try:
                app._append_transcript_raw(f"> {seed}")  # type: ignore[attr-defined]
            except Exception:
                pass

            if is_exit_command(seed):
                transcript.write("Exiting.")
                try:
                    app._append_transcript_raw("Exiting.")  # type: ignore[attr-defined]
                except Exception:
                    pass
                app.exit(0)
                return

            result = await asyncio.to_thread(process_line, seed)
            after_each_prompt()
            app._refresh_context()  # type: ignore[attr-defined]
            app._update_prompt()  # type: ignore[attr-defined]
            for part in renderables_for_result(result):
                transcript.write(part)

            try:
                app._append_transcript_raw(">>>>>>>>>")  # type: ignore[attr-defined]
                result_text = (result or "").rstrip()
                if result_text:
                    for out_line in result_text.splitlines():
                        app._append_transcript_raw(out_line)  # type: ignore[attr-defined]
                app._append_transcript_raw("")  # type: ignore[attr-defined]
            except Exception:
                pass

        app.call_later(_seed_after_ready)

    try:
        result = app.run()
    except SystemExit as exc:
        # Some Textual versions/drivers may raise SystemExit. Convert to a return
        # code so upstream cleanup logic still runs.
        code = getattr(exc, "code", 0)
        try:
            return int(0 if code is None else code)
        except Exception:
            return 0
    try:
        return int(0 if result is None else result)
    except Exception:
        return 0

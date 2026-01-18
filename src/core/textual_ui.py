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
    config: TextualReplConfig | None = None,
) -> int:
    """Run a full-screen Textual REPL.

    This UI is single-threaded (as terminals should be). Long-running `process_line`
    work is executed in a background thread so input and other panes remain responsive.
    """

    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, Input, Static, RichLog
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

        #transcript { height: 1fr; }
        #prompt_line { height: auto; }

        #context { height: 1fr; }
        """

        BINDINGS = [
            ("ctrl+c", "quit", "Quit"),
        ]

        def __init__(self) -> None:
            super().__init__()
            self._context_last_version: int = -1
            self._did_init_session: bool = False

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="left"):
                    yield RichLog(id="transcript", wrap=True)
                    yield Static("", id="prompt_line")
                    yield Input(placeholder="> ", id="command")
                with Vertical(id="right"):
                    yield Static("Context", id="context_title")
                    yield RichLog(id="context", wrap=True)
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#command", Input).focus()
            # Ensure the system prompt (and other session-start context) is always
            # injected even before the user submits anything.
            if not self._did_init_session:
                try:
                    before_first_prompt()
                finally:
                    self._did_init_session = True

            self._refresh_context()
            self._update_prompt()
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

            self._context_last_version = version
            try:
                snapshot = context_snapshot() or ""
            except Exception:
                snapshot = "(error rendering context)"

            view = self.query_one("#context", RichLog)
            view.clear()
            # Write line-by-line to preserve wrapping behavior.
            for line in snapshot.splitlines() or [""]:
                view.write(line)

        async def on_input_submitted(self, event: Input.Submitted) -> None:
            line = (event.value or "").strip()
            event.input.value = ""
            if not line:
                return

            transcript = self.query_one("#transcript", RichLog)
            transcript.write(f"> {line}")

            if is_exit_command(line):
                transcript.write("Exiting.")
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
            transcript = app.query_one("#transcript", RichLog)
            transcript.write(f"> {seed}")

            if is_exit_command(seed):
                transcript.write("Exiting.")
                app.exit(0)
                return

            result = await asyncio.to_thread(process_line, seed)
            after_each_prompt()
            app._refresh_context()  # type: ignore[attr-defined]
            app._update_prompt()  # type: ignore[attr-defined]
            for part in renderables_for_result(result):
                transcript.write(part)

        app.call_later(_seed_after_ready)

    return int(app.run())

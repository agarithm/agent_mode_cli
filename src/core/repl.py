from __future__ import annotations

from typing import Callable, Optional

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.text import Text
except Exception:
    Console = None  # type: ignore
    Markdown = None  # type: ignore
    Text = None  # type: ignore


def is_exit_command(line: str) -> bool:
    return (line or "").strip().lower() in ("exit", "quit", "q")


def renderables_for_result(result: str):
    """Return a list of Rich renderables/strings for a REPL result."""

    text = (result or "").rstrip()
    if Text is None:
        separator = ">>>>>>>>>"
    else:
        separator = Text(">>>>>>>>>", style="bold cyan")

    if Markdown is None:
        return [separator, text, ""]

    try:
        return [separator, Markdown(text or ""), ""]
    except Exception:
        return [separator, text, ""]


def run_repl(
    *,
    initial_line: Optional[str],
    before_first_prompt: Callable[[], None],
    process_line: Callable[[str], str],
    after_each_prompt: Callable[[], None],
    prompt_provider: Optional[Callable[[], str]] = None,
) -> int:
    console = Console() if Console is not None else None

    def _prompt() -> str:
        if prompt_provider is None:
            return "> "
        try:
            value = prompt_provider() or "> "
        except Exception:
            value = "> "
        return f"{value}"

    def _render_result(result: str) -> None:
        parts = list(renderables_for_result(result))
        if console is None:
            for part in parts:
                print(str(part))
            return
        for part in parts:
            console.print(part)

    try:
        before_first_prompt()

        if initial_line is not None and initial_line.strip():
            try:
                result = process_line(initial_line.strip())
            finally:
                after_each_prompt()
            _render_result(result)

        while True:
            prompt_text = _prompt()
            if console is None:
                line = input(prompt_text)
            else:
                console.print(prompt_text, style="bold cyan", end="", markup=False)
                line = input()
            if not line.strip():
                continue
            if is_exit_command(line):
                print("Exiting.")
                break
            try:
                result = process_line(line)
            finally:
                after_each_prompt()
            _render_result(result)
        return 0
    except EOFError:
        print("\nInterrupted – exiting.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted – exiting.")
        return 0

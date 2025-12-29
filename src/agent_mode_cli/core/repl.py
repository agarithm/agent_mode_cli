from __future__ import annotations

from typing import Callable, Optional

try:
    from rich.console import Console
    from rich.markdown import Markdown
except Exception:
    Console = None  # type: ignore
    Markdown = None  # type: ignore


def run_repl(
    *,
    initial_line: Optional[str],
    before_first_prompt: Callable[[], None],
    process_line: Callable[[str], str],
    after_each_prompt: Callable[[], None],
) -> int:
    console = Console() if Console is not None else None

    def _render_result(result: str) -> None:
        text = (result or "").rstrip()
        if console is None or Markdown is None:
            print(f">>> {text}\n")
            return

        console.print("[bold cyan]>>>[/bold cyan]", end=" ")
        try:
            markdown = Markdown(text or "")
            console.print(markdown)
        except Exception:
            console.print(text)
        finally:
            console.print()

    try:
        before_first_prompt()

        if initial_line is not None and initial_line.strip():
            try:
                result = process_line(initial_line.strip())
            finally:
                after_each_prompt()
            _render_result(result)

        while True:
            line = input("> ")
            if not line.strip():
                continue
            if line.strip().lower() in ("exit", "quit", "q"):
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

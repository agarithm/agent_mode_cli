from __future__ import annotations

from typing import Callable, Optional


def run_repl(
    *,
    initial_line: Optional[str],
    before_first_prompt: Callable[[], None],
    process_line: Callable[[str], str],
    after_each_prompt: Callable[[], None],
) -> int:
    try:
        before_first_prompt()

        if initial_line is not None and initial_line.strip():
            try:
                result = process_line(initial_line.strip())
            finally:
                after_each_prompt()
            print(f">>> {result}\n")

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
            print(f">>> {result}\n")
        return 0
    except EOFError:
        print("\nInterrupted – exiting.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted – exiting.")
        return 0

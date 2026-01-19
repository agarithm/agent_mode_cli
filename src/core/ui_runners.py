from __future__ import annotations

import sys
from typing import Any

from core.repl import run_repl
from core.repl_callbacks import ReplCallbacks


def run_inline(callbacks: ReplCallbacks) -> int:
    return run_repl(
        initial_line=callbacks.initial_line,
        before_first_prompt=callbacks.before_first_prompt,
        process_line=callbacks.process_line,
        after_each_prompt=callbacks.after_each_prompt,
        prompt_provider=callbacks.prompt_provider,
    )


def run_fullscreen(callbacks: ReplCallbacks) -> int:
    """Run the full-screen UI if available; otherwise fall back to inline."""

    try:
        from core.textual_ui import run_textual_repl
        from core.textual_confirm import TextualPromptBridge
        from core.prompting import set_prompt_backend
    except ModuleNotFoundError:
        print("warning: full-screen UI unavailable; falling back to --inline", file=sys.stderr)
        return run_inline(callbacks)

    prompt_bridge = TextualPromptBridge()

    def _on_ready(app: Any) -> None:
        try:
            prompt_bridge.attach_app(app)
        except Exception:
            pass
        try:
            set_prompt_backend(prompt_bridge)
        except Exception:
            pass
        if callbacks.on_app_ready is not None:
            try:
                callbacks.on_app_ready(app)
            except Exception:
                pass

    return run_textual_repl(
        initial_line=callbacks.initial_line,
        before_first_prompt=callbacks.before_first_prompt,
        process_line=callbacks.process_line,
        after_each_prompt=callbacks.after_each_prompt,
        prompt_provider=callbacks.prompt_provider,
        context_version=callbacks.context_version,
        context_snapshot=callbacks.context_snapshot,
        context_delta=callbacks.context_delta,
        on_app_ready=_on_ready,
    )

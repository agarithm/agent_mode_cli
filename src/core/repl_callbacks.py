from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class ReplCallbacks:
    initial_line: Optional[str]
    before_first_prompt: Callable[[], None]
    process_line: Callable[[str], str]
    after_each_prompt: Callable[[], None]
    prompt_provider: Optional[Callable[[], str]] = None

    # Optional hooks for richer UIs.
    context_version: Optional[Callable[[], int]] = None
    context_snapshot: Optional[Callable[[], str]] = None
    context_delta: Optional[Callable[[int], str]] = None
    on_app_ready: Optional[Callable[[object], None]] = None

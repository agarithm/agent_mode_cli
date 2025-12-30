from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.universal_context import ChatMessage


@dataclass(frozen=True)
class ToolCallInfo:
    name: str
    arguments: Dict[str, Any]
    raw: Any
    call_id: Optional[str] = None


@dataclass(frozen=True)
class ParseResult:
    tool_calls: Sequence[ToolCallInfo]
    context_entries: Sequence[ChatMessage]
    final_text: Optional[str]


def safe_json_loads(value: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if default is None:
        default = {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else default
    except Exception:
        return default


def process_line_with_tools(
    line: str,
    *,
    debug: bool,
    append_context: Callable[[ChatMessage], None],
    call_model: Callable[[], Any],
    parse_response: Callable[[Any], ParseResult],
    execute_tool_call: Callable[[ToolCallInfo], Tuple[Sequence[ChatMessage], Optional[str]]],
    max_tool_iterations: int = 100,
    max_tool_seconds: Optional[float] = None,
) -> str:
    append_context(ChatMessage(role="user", content=line))
    printed_progress = False

    if max_tool_iterations <= 0:
        max_tool_iterations = 1
    start = time.monotonic()
    iterations = 0

    while True:
        iterations += 1
        if iterations > max_tool_iterations:
            if printed_progress:
                print()
            error_text = (
                "error: exceeded max tool iterations "
                f"({max_tool_iterations}). The model may be stuck in a tool loop."
            )
            append_context(ChatMessage(role="assistant", content=error_text))
            return error_text

        if max_tool_seconds is not None and max_tool_seconds > 0:
            elapsed = time.monotonic() - start
            if elapsed > max_tool_seconds:
                if printed_progress:
                    print()
                error_text = (
                    "error: exceeded tool-loop time budget "
                    f"({max_tool_seconds:.1f}s). The model may be stuck in a tool loop."
                )
                append_context(ChatMessage(role="assistant", content=error_text))
                return error_text

        if not debug:
            print(".", end="", flush=True)
            printed_progress = True

        response = call_model()
        parsed = parse_response(response)
        for entry in parsed.context_entries:
            append_context(entry)

        if not parsed.tool_calls:
            if printed_progress:
                print()
            return parsed.final_text or "(no content)"

        cancelled_message: Optional[str] = None
        for call in parsed.tool_calls:
            entries, cancelled_message = execute_tool_call(call)
            for entry in entries:
                append_context(entry)
            if cancelled_message:
                break

        if cancelled_message:
            if printed_progress:
                print()
            append_context(ChatMessage(role="assistant", content=cancelled_message))
            return cancelled_message

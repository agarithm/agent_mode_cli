from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ToolCallInfo:
    name: str
    arguments: Dict[str, Any]
    raw: Any
    call_id: Optional[str] = None


@dataclass(frozen=True)
class ParseResult:
    tool_calls: Sequence[ToolCallInfo]
    context_entries: Sequence[Any]
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
    append_context: Callable[[Any], None],
    call_model: Callable[[], Any],
    parse_response: Callable[[Any], ParseResult],
    execute_tool_call: Callable[[ToolCallInfo], Tuple[Sequence[Any], Optional[str]]],
) -> str:
    append_context({"role": "user", "content": line})
    printed_progress = False

    while True:
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
            append_context({"role": "assistant", "content": cancelled_message})
            return cancelled_message

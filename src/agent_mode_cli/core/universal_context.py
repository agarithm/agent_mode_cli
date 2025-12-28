from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass(frozen=True)
class ChatMessage:
    """Universal internal context message.

    This is the platform-neutral representation we keep internally.
    Provider adapters translate these messages into their required schemas.

    Fields:
    - role: "system" | "user" | "assistant" | "tool" (extensible)
    - content: textual content for the message/tool output
    - tool_name: for role=="tool" messages, the tool/function name
    - tool_call_id: for role=="tool" messages, which tool-call this output satisfies
    - tool_calls: for assistant messages that include tool calls
    """

    role: str
    content: str
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_calls: Optional[Sequence[ToolCall]] = None


class UniversalContext:
    """Mutable container for universal internal chat context."""

    def __init__(self) -> None:
        self._messages: List[ChatMessage] = []

    def __len__(self) -> int:
        return len(self._messages)

    @property
    def messages(self) -> List[ChatMessage]:
        return self._messages

    def append(self, message: ChatMessage, *, debug: bool = False) -> None:
        role = (message.role or "").strip()
        if not role:
            raise ValueError("context messages must include a role")
        # Drop truly empty messages (unless they carry tool calls).
        if (message.content or "").strip() == "" and not message.tool_calls:
            if debug:
                print("[debug] skipped appending empty content message")
            return
        self._messages.append(message)

    def extend(self, messages: Iterable[ChatMessage], *, debug: bool = False) -> None:
        for msg in messages:
            self.append(msg, debug=debug)


def to_openai_responses_input(messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    """Translate universal messages into OpenAI Responses API input items.

    This returns dict-only items to avoid relying on SDK-specific object classes.
    """

    items: List[Dict[str, Any]] = []

    for msg in messages:
        role = (msg.role or "").strip()

        if role in {"system", "user", "assistant"}:
            if msg.content.strip():
                items.append({"role": role, "content": msg.content})

            if msg.tool_calls:
                for call in msg.tool_calls:
                    payload: Dict[str, Any] = {
                        "type": "function_call",
                        "name": call.name,
                        "arguments": json.dumps(call.arguments or {}),
                    }
                    if call.call_id:
                        payload["call_id"] = call.call_id
                    items.append(payload)
            continue

        if role == "tool":
            payload = {
                "type": "function_call_output",
                "output": (msg.content or "").strip() or "(no output)",
            }
            if msg.tool_call_id:
                payload["call_id"] = msg.tool_call_id
            items.append(payload)
            continue

        # Unknown role: keep text as assistant content.
        if msg.content.strip():
            items.append({"role": "assistant", "content": msg.content})

    return items


def to_ollama_chat_messages(messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    """Translate universal messages into Ollama chat messages."""

    out: List[Dict[str, Any]] = []

    for msg in messages:
        role = (msg.role or "").strip() or "assistant"
        entry: Dict[str, Any] = {"role": role, "content": msg.content}
        if role == "tool" and msg.tool_name:
            entry["tool_name"] = msg.tool_name
        if msg.tool_calls:
            entry["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments or {}}}
                for call in msg.tool_calls
            ]
        out.append(entry)
    return out

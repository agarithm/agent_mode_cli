from __future__ import annotations

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


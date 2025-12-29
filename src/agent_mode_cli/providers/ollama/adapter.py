from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from agent_mode_cli.core.agent_loop import ParseResult, ToolCallInfo
from agent_mode_cli.core.universal_context import ChatMessage, ToolCall, UniversalContext


def _to_ollama_chat_messages(messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
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


@dataclass(frozen=True)
class OllamaProviderAdapter:
    client: Any

    def _normalize_arguments(self, arguments: Any) -> Dict[str, Any]:
        if arguments is None:
            return {}
        if isinstance(arguments, Mapping):
            return dict(arguments)
        if hasattr(arguments, "model_dump"):
            return arguments.model_dump()  # type: ignore[no-any-return]
        if hasattr(arguments, "dict"):
            return arguments.dict()  # type: ignore[no-any-return]
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                return {"value": arguments}
        try:
            return dict(arguments)  # type: ignore[arg-type]
        except Exception:
            return {"value": arguments}

    def _debug_dump_context(self, context: UniversalContext) -> None:
        print(f"[debug] calling model with context of {len(context)} messages")
        for idx, msg in enumerate(context.messages):
            role = msg.role or "?"
            content = msg.content or ""
            preview = content if isinstance(content, str) else json.dumps(content)
            if len(preview) > 80:
                preview = preview[:77] + "..."
            extra = ""
            if msg.tool_calls:
                extra = f" tool_calls={len(msg.tool_calls)}"
            if msg.tool_name:
                extra = f" tool_name={msg.tool_name}"
            print(f"[debug]   {idx:02d} {role}: {preview}{extra}")

    def call_model(self, *, model: str, tools: Sequence[Dict[str, Any]], context: UniversalContext, debug: bool) -> Any:
        if debug:
            self._debug_dump_context(context)
        try:
            return self.client.chat(model=model, messages=_to_ollama_chat_messages(context.messages), tools=list(tools))
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def parse_response(self, response: Any, *, debug: bool) -> ParseResult:
        message = getattr(response, "message", None)
        if message is None:
            error_text = "error: empty response from model"
            return ParseResult(tool_calls=[], context_entries=[ChatMessage(role="assistant", content=error_text)], final_text=error_text)

        assistant_content = (getattr(message, "content", "") or "").strip()
        raw_tool_calls = list(getattr(message, "tool_calls", None) or [])

        tool_calls: list[ToolCallInfo] = []
        assistant_tool_calls: list[ToolCall] = []
        if raw_tool_calls:
            for idx, call in enumerate(raw_tool_calls):
                function = getattr(call, "function", None)
                name = getattr(function, "name", None)
                if not name:
                    continue
                arguments = self._normalize_arguments(getattr(function, "arguments", None))
                call_id = f"call_{idx}"  # Ollama tool calls do not expose a stable call_id
                tool_calls.append(ToolCallInfo(name=name, arguments=arguments, raw=None, call_id=call_id))
                assistant_tool_calls.append(ToolCall(name=name, arguments=arguments, call_id=call_id))

        assistant_entry = ChatMessage(
            role=getattr(message, "role", None) or "assistant",
            content=assistant_content,
            tool_calls=assistant_tool_calls or None,
        )

        final_text = assistant_content or "(no content)"
        return ParseResult(tool_calls=tool_calls, context_entries=[assistant_entry], final_text=final_text)

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from core.agent_loop import ParseResult, ToolCallInfo, safe_json_loads
from core.universal_context import ChatMessage, ToolCall, UniversalContext
from providers.base import ProviderRateLimitError


def _to_openai_responses_input(messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    """Translate universal messages into OpenAI Responses API input items.

    Returns dict-only items to avoid relying on SDK-specific object classes.
    """

    items: List[Dict[str, Any]] = []

    for msg in messages:
        role = (msg.role or "").strip()

        if role in {"system", "user", "assistant"}:
            if (msg.content or "").strip():
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
            payload: Dict[str, Any] = {
                "type": "function_call_output",
                "output": (msg.content or "").strip() or "(no output)",
            }
            if msg.tool_call_id:
                payload["call_id"] = msg.tool_call_id
            items.append(payload)
            continue

        # Unknown role: keep text as assistant content.
        if (msg.content or "").strip():
            items.append({"role": "assistant", "content": msg.content})

    return items


@dataclass(frozen=True)
class OpenAIProviderAdapter:
    client: Any

    def list_models(self, *, debug: bool) -> Sequence[str]:
        if debug:
            print("[debug] listing OpenAI models")
        result = self.client.models.list()
        data = getattr(result, "data", None)
        if not data:
            return []
        models: list[str] = []
        for item in data:
            mid = getattr(item, "id", None)
            if isinstance(mid, str) and mid.strip():
                models.append(mid.strip())
        return sorted(set(models))

    def call_model(self, *, model: str, tools: Sequence[Dict[str, Any]], context: UniversalContext, debug: bool) -> Any:
        if debug:
            print(f"[debug] calling model with context of {len(context)} messages")
        try:
            return self.client.responses.create(model=model, tools=list(tools), input=_to_openai_responses_input(context.messages))
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status is None:
                status = getattr(getattr(exc, "response", None), "status_code", None)
            retry_after = None
            if status == 429:
                headers = getattr(exc, "headers", None)
                if headers is None:
                    resp = getattr(exc, "response", None)
                    headers = getattr(resp, "headers", None) if resp is not None else None
                if headers:
                    retry_after_header = headers.get("retry-after")
                    if retry_after_header is not None:
                        try:
                            retry_after = float(retry_after_header)
                        except (TypeError, ValueError):
                            retry_after = None
                raise ProviderRateLimitError('OpenAI rate limit hit', provider='openai', retry_after=retry_after) from exc
            raise

    def _stringify_reasoning(self, item: Any) -> str:
        pieces: list[str] = []
        for chunk in getattr(item, "content", []) or []:
            text = getattr(chunk, "text", None)
            if text:
                pieces.append(text)
        fallback = getattr(item, "text", None)
        if fallback:
            pieces.append(fallback)
        return "\n".join(pieces).strip()

    def parse_response(self, response: Any, *, debug: bool) -> ParseResult:
        output = getattr(response, "output", []) or []
        tool_calls: list[ToolCallInfo] = []

        context_entries: list[ChatMessage] = []
        if output and getattr(output[0], "type", None) == "reasoning":
            reasoning_text = self._stringify_reasoning(output[0])
            if reasoning_text:
                context_entries.append(ChatMessage(role="assistant", content=f"[reasoning]\n{reasoning_text}"))
                if debug:
                    print("[debug] reasoning detail captured as text")

        calls_for_assistant: list[ToolCall] = []
        for idx, item in enumerate(output):
            if getattr(item, "type", None) != "function_call":
                continue
            args = safe_json_loads(getattr(item, "arguments", "") or "{}")
            call_id = getattr(item, "call_id", None) or f"call_{idx}"
            tool_calls.append(ToolCallInfo(name=item.name, arguments=args, raw=None, call_id=call_id))
            calls_for_assistant.append(ToolCall(name=item.name, arguments=args, call_id=call_id))

        if calls_for_assistant:
            context_entries.append(ChatMessage(role="assistant", content="", tool_calls=calls_for_assistant))

        final_text = getattr(response, "output_text", None)
        if not tool_calls and final_text:
            context_entries.append(ChatMessage(role="assistant", content=final_text))

        return ParseResult(tool_calls=tool_calls, context_entries=context_entries, final_text=final_text)

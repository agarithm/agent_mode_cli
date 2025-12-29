from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import httpx

from agent_mode_cli.core.agent_loop import ParseResult, ToolCallInfo, safe_json_loads
from agent_mode_cli.core.universal_context import ChatMessage, ToolCall, UniversalContext
from agent_mode_cli.providers.base import ProviderRateLimitError
from agent_mode_cli.providers.copilot.runtime import maybe_resolve_model_alias, resolve_gemini_model


def _to_chat_completions_messages(messages: Sequence[ChatMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        role = (msg.role or "").strip() or "assistant"

        if role == "tool":
            payload: Dict[str, Any] = {
                "role": "tool",
                "content": (msg.content or ""),
            }
            if msg.tool_call_id:
                payload["tool_call_id"] = msg.tool_call_id
            # name is not required by the OpenAI-compatible schema, but harmless if present.
            if msg.tool_name:
                payload["name"] = msg.tool_name
            out.append(payload)
            continue

        payload = {"role": role, "content": (msg.content or "")}
        if msg.tool_calls and role == "assistant":
            tool_calls: List[Dict[str, Any]] = []
            for idx, call in enumerate(msg.tool_calls):
                tool_calls.append(
                    {
                        "id": call.call_id or f"call_{idx}",
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments or {}),
                        },
                    }
                )
            payload["tool_calls"] = tool_calls
        out.append(payload)
    return out


@dataclass(frozen=True)
class CopilotProviderAdapter:
    """Provider adapter backed by GitHub Models inference (OpenAI-compatible chat completions)."""

    client: httpx.Client

    def list_models(self, *, debug: bool) -> Sequence[str]:
        if debug:
            print("[debug] listing GitHub Models catalog")
        resp = self.client.get("/catalog/models")
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        models: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            mid = (item.get("id") or "").strip()
            if mid:
                models.append(mid)
        return sorted(set(models))

    def call_model(self, *, model: str, tools: Sequence[Dict[str, Any]], context: UniversalContext, debug: bool) -> Any:
        initial_model = maybe_resolve_model_alias(model, self.client)
        if debug and initial_model != model:
            print(f"[debug] resolved model alias '{model}' -> '{initial_model}'")

        def _request(chosen_model: str) -> Any:
            payload: Dict[str, Any] = {
                "model": chosen_model,
                "messages": _to_chat_completions_messages(context.messages),
            }
            if tools:
                payload["tools"] = list(tools)
                payload["tool_choice"] = "auto"
            if debug:
                print(f"[debug] github models request: model={chosen_model} messages={len(context)} tools={len(tools)}")
            resp = self.client.post("/inference/chat/completions", json=payload)
            resp.raise_for_status()
            return resp.json()

        try:
            return _request(initial_model)
        except httpx.HTTPStatusError as exc:
            retry_after: Optional[float] = None
            try:
                body = exc.response.json()
            except Exception:
                body = None

            error_code = None
            if isinstance(body, dict):
                err = body.get("error")
                if isinstance(err, dict):
                    error_code = (err.get("code") or "").strip().lower() or None
                    retry_after_value = err.get("retry_after") if err else None
                    if retry_after_value is not None:
                        try:
                            retry_after = float(retry_after_value)
                        except (TypeError, ValueError):
                            retry_after = None

            if exc.response.status_code == 404 and error_code == "unknown_model":
                recovered_model = resolve_gemini_model(self.client)
                if debug:
                    print(f"[debug] unknown_model for '{initial_model}', retrying with '{recovered_model}'")
                try:
                    return _request(recovered_model)
                except Exception:
                    pass

            if exc.response.status_code == 429:
                raise ProviderRateLimitError(
                    "GitHub Models rate limit hit",
                    provider="copilot",
                    retry_after=retry_after,
                ) from exc

            text = (exc.response.text or "").strip()
            raise RuntimeError(f"GitHub Models inference failed: HTTP {exc.response.status_code} {text}") from exc
        except Exception as exc:
            raise RuntimeError(str(exc) or "GitHub Models inference failed") from exc

    def parse_response(self, response: Any, *, debug: bool) -> ParseResult:
        if not isinstance(response, dict):
            error_text = "error: unexpected response type from provider"
            return ParseResult(tool_calls=[], context_entries=[ChatMessage(role="assistant", content=error_text)], final_text=error_text)

        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            error_text = "error: empty response from model"
            return ParseResult(tool_calls=[], context_entries=[ChatMessage(role="assistant", content=error_text)], final_text=error_text)

        message = (choices[0] or {}).get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            error_text = "error: malformed response from model"
            return ParseResult(tool_calls=[], context_entries=[ChatMessage(role="assistant", content=error_text)], final_text=error_text)

        content = str(message.get("content") or "").strip()

        tool_calls_raw = message.get("tool_calls")
        tool_calls: List[ToolCallInfo] = []
        assistant_tool_calls: List[ToolCall] = []

        if isinstance(tool_calls_raw, list) and tool_calls_raw:
            for idx, tc in enumerate(tool_calls_raw):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name") or "").strip()
                if not name:
                    continue
                args_text = fn.get("arguments")
                args = safe_json_loads(args_text if isinstance(args_text, str) else "{}")
                call_id = str(tc.get("id") or "").strip() or f"call_{idx}"
                tool_calls.append(ToolCallInfo(name=name, arguments=args, raw=tc, call_id=call_id))
                assistant_tool_calls.append(ToolCall(name=name, arguments=args, call_id=call_id))

        assistant_entry = ChatMessage(
            role=str(message.get("role") or "assistant"),
            content=content,
            tool_calls=assistant_tool_calls or None,
        )

        final_text: Optional[str] = content if content else None
        return ParseResult(tool_calls=tool_calls, context_entries=[assistant_entry], final_text=final_text)

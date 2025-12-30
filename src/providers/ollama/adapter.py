from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from core.agent_loop import ParseResult, ToolCallInfo
from core.universal_context import ChatMessage, ToolCall, UniversalContext


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

    def list_models(self, *, debug: bool) -> Sequence[str]:
        if debug:
            print("[debug] listing Ollama models")
        result = self.client.list()
        models: list[str] = []
        if hasattr(result, "models"):
            for item in getattr(result, "models") or []:
                name = getattr(item, "model", None) or getattr(item, "name", None)
                if isinstance(name, str) and name.strip():
                    models.append(name.strip())
            return sorted(set(models))

        if isinstance(result, dict):
            items = result.get("models")
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        name = (item.get("name") or "").strip()
                        if name:
                            models.append(name)
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    name = (item.get("name") or "").strip()
                    if name:
                        models.append(name)
                elif isinstance(item, str) and item.strip():
                    models.append(item.strip())
        return sorted(set(models))

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

    def _extract_tool_call_specs_from_json(self, payload: Any) -> List[tuple[str, Dict[str, Any]]]:
        specs: List[tuple[str, Dict[str, Any]]] = []
        seen: set[tuple[str, str]] = set()

        def _walk(node: Any) -> None:
            if isinstance(node, Mapping):
                tool_calls = node.get("tool_calls")
                if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes, bytearray)):
                    for item in tool_calls:
                        _walk(item)

                function_obj = node.get("function")
                if isinstance(function_obj, Mapping):
                    name = function_obj.get("name")
                    arguments = function_obj.get("arguments")
                    if isinstance(name, str) and name.strip() and arguments is not None:
                        normalized = self._normalize_arguments(arguments)
                        signature = (name.strip(), json.dumps(normalized, sort_keys=True, default=str))
                        if signature not in seen:
                            seen.add(signature)
                            specs.append((name.strip(), normalized))

                name = node.get("name")
                arguments = node.get("arguments")
                if isinstance(name, str) and name.strip() and arguments is not None:
                    normalized = self._normalize_arguments(arguments)
                    signature = (name.strip(), json.dumps(normalized, sort_keys=True, default=str))
                    if signature not in seen:
                        seen.add(signature)
                        specs.append((name.strip(), normalized))

                call_obj = node.get("call")
                if isinstance(call_obj, Mapping):
                    _walk(call_obj)

            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
                for item in node:
                    _walk(item)

        _walk(payload)
        return specs

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

        parsed_json: Any = None
        extracted_text: str | None = None

        # Detect if response is JSON (models like qwen2.5-coder return JSON-formatted responses)
        if assistant_content and assistant_content.startswith(("{", "[")):
            try:
                parsed_json = json.loads(assistant_content)
            except (json.JSONDecodeError, ValueError):
                parsed_json = None

        text_fields = ("text", "response", "content", "message", "output", "answer")
        if isinstance(parsed_json, dict):
            for key in text_fields:
                value = parsed_json.get(key)
                if isinstance(value, str) and value.strip():
                    extracted_text = value.strip()
                    if debug:
                        print(f"[ollama] Extracted text from JSON response field '{key}'", file=__import__("sys").stderr)
                    assistant_content = extracted_text
                    break
            else:
                if debug:
                    print(f"[ollama] JSON response has no recognized text field: {list(parsed_json.keys())}", file=__import__("sys").stderr)
        elif isinstance(parsed_json, list) and parsed_json:
            for item in parsed_json:
                if not isinstance(item, Mapping):
                    continue
                for key in text_fields:
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        extracted_text = value.strip()
                        if debug:
                            print(f"[ollama] Extracted text from JSON array item field '{key}'", file=__import__("sys").stderr)
                        assistant_content = extracted_text
                        break
                if extracted_text:
                    break

        raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
        fallback_tool_specs: List[tuple[str, Dict[str, Any]]] = []
        if not raw_tool_calls and parsed_json is not None:
            fallback_tool_specs = self._extract_tool_call_specs_from_json(parsed_json)
            if fallback_tool_specs and debug:
                print("[ollama] Detected tool call(s) embedded in JSON content", file=__import__("sys").stderr)

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
        elif fallback_tool_specs:
            for idx, (name, arguments) in enumerate(fallback_tool_specs):
                call_id = f"call_json_{idx}"
                tool_calls.append(ToolCallInfo(name=name, arguments=arguments, raw=None, call_id=call_id))
                assistant_tool_calls.append(ToolCall(name=name, arguments=arguments, call_id=call_id))
            if extracted_text is None:
                assistant_content = ""

        assistant_entry = ChatMessage(
            role=getattr(message, "role", None) or "assistant",
            content=assistant_content,
            tool_calls=assistant_tool_calls or None,
        )

        final_text = assistant_content or "(no content)"
        return ParseResult(tool_calls=tool_calls, context_entries=[assistant_entry], final_text=final_text)

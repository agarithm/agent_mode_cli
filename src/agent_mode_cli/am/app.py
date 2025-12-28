from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI
try:  # openai>=1.x
    from openai import OpenAIError  # type: ignore
except Exception:  # pragma: no cover - defensive
    OpenAIError = Exception  # type: ignore

from agent_mode_cli.core.bash_tool import bash_command
from agent_mode_cli.core.agent_loop import ParseResult, ToolCallInfo, process_line_with_tools, safe_json_loads
from agent_mode_cli.core.confirm import ConfirmState, prompt_for_confirmation, requires_confirmation
from agent_mode_cli.core.prompt_file import load_user_prompt
from agent_mode_cli.core.repl import run_repl
from agent_mode_cli.core.system_prompt import build_internal_system_prompt
from agent_mode_cli.core.tool_specs import build_openai_tools


DEBUG = os.getenv("AM_DEBUG", "").lower() in ("1", "true", "yes", "on")
DEFAULT_MODEL = os.getenv("AM_MODEL", "gpt-5.1-codex")

INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("AM")
def main(argv: list[str] | None = None) -> int:
    global DEBUG, DEFAULT_MODEL

    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print("usage: am <prompt...>")
        print("\nAgent-mode CLI backed by OpenAI.\n")
        print("env:")
        print("  OPENAI_API_KEY  required")
        print("  AM_MODEL        optional (default: gpt-5.1-codex)")
        print("  AM_DEBUG        optional (1/true enables debug)")
        return 0
    if argv and argv[0] == "--version":
        from agent_mode_cli import __version__

        print(__version__)
        return 0
    initial_line = " ".join(argv) if argv else None

    try:
        client = OpenAI()
    except OpenAIError as exc:
        message = str(exc).strip() or "OpenAI client initialization failed"
        if "api_key" in message.lower() or "OPENAI_API_KEY" in message:
            print("error: OPENAI_API_KEY is not set", file=sys.stderr)
        else:
            print(f"error: {message}", file=sys.stderr)
        return 1
    context: list[Any] = []
    tools = build_openai_tools("AM")
    state = ConfirmState(approve_all=False, debug=DEBUG)

    def set_debug_command(enabled: bool) -> str:
        nonlocal state
        global DEBUG
        DEBUG = bool(enabled)
        os.environ["AM_DEBUG"] = "1" if DEBUG else "0"
        state.debug = DEBUG
        return f"AM_DEBUG {'enabled' if DEBUG else 'disabled'}."

    def set_model_command(model: str) -> str:
        global DEFAULT_MODEL
        model = (model or "").strip()
        if not model:
            return "error: model is required"
        DEFAULT_MODEL = model
        os.environ["AM_MODEL"] = model
        return f"AM_MODEL set to {model}."

    TOOL_FUNCTIONS = {
        "set_debug": set_debug_command,
        "set_model": set_model_command,
        "bash": lambda command="": bash_command(command),
    }

    def append_context(item: Any) -> None:
        if isinstance(item, dict):
            if not item:
                if DEBUG:
                    print("[debug] skipped appending empty dict to context")
                return
            content = item.get("content")
            extra_keys = set(item.keys()) - {"role", "content"}
            if isinstance(content, str) and not content.strip() and not extra_keys:
                if DEBUG:
                    print("[debug] skipped appending empty content message")
                return
        context.append(item)

    def extend_context(items: list[Any]) -> None:
        for entry in items:
            append_context(entry)

    def call_model() -> Any:
        if DEBUG:
            print(f"[debug] calling model with context of {len(context)} messages")
        return client.responses.create(model=DEFAULT_MODEL, tools=tools, input=context)

    def stringify_reasoning(item: Any) -> str:
        pieces: list[str] = []
        for chunk in getattr(item, "content", []) or []:
            text = getattr(chunk, "text", None)
            if text:
                pieces.append(text)
        fallback = getattr(item, "text", None)
        if fallback:
            pieces.append(fallback)
        return "\n".join(pieces).strip()

    def tool_call(item: Any, args: dict[str, Any] | None = None) -> list[Any]:
        handler = TOOL_FUNCTIONS.get(item.name)
        output_text = None
        if args is None:
            try:
                args = json.loads(item.arguments or "{}")
            except json.JSONDecodeError as exc:
                args = {}
                handler = None
                output_text = f"error: could not parse arguments ({exc})"
        if handler is None:
            if output_text is None:
                output_text = f"error: unknown tool '{item.name}'"
        else:
            try:
                result = handler(**args)
            except TypeError as exc:
                result = f"error: invalid arguments - {exc}"
            except Exception as exc:
                result = f"error: {exc}"
            output_text = (result or "").strip() or "(no output)"
        return [
            item,
            {"type": "function_call_output", "call_id": item.call_id, "output": output_text},
        ]

    def handle_tools(response: Any) -> tuple[bool, bool, str]:
        has_function_call = any(item.type == "function_call" for item in response.output)
        if response.output and response.output[0].type == "reasoning":
            reasoning_item = response.output[0]
            if has_function_call:
                append_context(reasoning_item)
                if DEBUG:
                    print("[debug] reasoning event stored for tool call")
            else:
                reasoning_text = stringify_reasoning(reasoning_item)
                if reasoning_text:
                    append_context({"role": "assistant", "content": f"[reasoning]\n{reasoning_text}"})
                    if DEBUG:
                        print("[debug] reasoning detail captured as text")
        osz = len(context)
        cancelled = False
        cancel_message = ""
        for item in response.output:
            if item.type != "function_call":
                continue
            if DEBUG:
                print(f"[debug] executing tool: {item.name}")
            try:
                args = json.loads(item.arguments or "{}") if item.arguments else {}
                if not isinstance(args, dict):
                    raise ValueError("tool arguments must be an object")
            except (json.JSONDecodeError, ValueError) as exc:
                extend_context(
                    [
                        item,
                        {"type": "function_call_output", "call_id": item.call_id, "output": f"error: unable to parse arguments ({exc})"},
                    ]
                )
                continue
            if requires_confirmation(item.name):
                if not prompt_for_confirmation(item.name, args, state):
                    cancel_message = f"Tool '{item.name}' execution cancelled by user."
                    extend_context(
                        [
                            item,
                            {"type": "function_call_output", "call_id": item.call_id, "output": "cancelled by user"},
                        ]
                    )
                    append_context({"role": "assistant", "content": cancel_message})
                    cancelled = True
                    break
            extend_context(tool_call(item, args=args))
        return len(context) != osz, cancelled, cancel_message

    def parse_response(response: Any) -> ParseResult:
        output = getattr(response, "output", []) or []
        tool_calls: List[ToolCallInfo] = []
        has_function_call = any(getattr(item, "type", None) == "function_call" for item in output)

        context_entries: List[Any] = []
        if output and getattr(output[0], "type", None) == "reasoning":
            reasoning_item = output[0]
            if has_function_call:
                context_entries.append(reasoning_item)
                if DEBUG:
                    print("[debug] reasoning event stored for tool call")
            else:
                reasoning_text = stringify_reasoning(reasoning_item)
                if reasoning_text:
                    context_entries.append({"role": "assistant", "content": f"[reasoning]\n{reasoning_text}"})
                    if DEBUG:
                        print("[debug] reasoning detail captured as text")

        for item in output:
            if getattr(item, "type", None) != "function_call":
                continue
            args = safe_json_loads(getattr(item, "arguments", "") or "{}")
            tool_calls.append(ToolCallInfo(name=item.name, arguments=args, raw=item, call_id=getattr(item, "call_id", None)))

        final_text = getattr(response, "output_text", None)
        if not tool_calls and final_text:
            context_entries.append({"role": "assistant", "content": final_text})

        return ParseResult(tool_calls=tool_calls, context_entries=context_entries, final_text=final_text)

    def execute_tool_call_info(call: ToolCallInfo) -> tuple[list[Any], str | None]:
        if DEBUG:
            print(f"[debug] executing tool: {call.name}")
        if requires_confirmation(call.name):
            if not prompt_for_confirmation(call.name, call.arguments, state):
                cancel_message = f"Tool '{call.name}' execution cancelled by user."
                return (
                    [
                        call.raw,
                        {"type": "function_call_output", "call_id": call.call_id, "output": "cancelled by user"},
                    ],
                    cancel_message,
                )

        handler = TOOL_FUNCTIONS.get(call.name)
        if handler is None:
            output_text = f"error: unknown tool '{call.name}'"
        else:
            try:
                output_text = (handler(**call.arguments) or "").strip() or "(no output)"
            except TypeError as exc:
                output_text = f"error: invalid arguments - {exc}"
            except Exception as exc:
                output_text = f"error: {exc}"
        return (
            [
                call.raw,
                {"type": "function_call_output", "call_id": call.call_id, "output": output_text},
            ],
            None,
        )

    def process(line: str) -> str:
        if DEBUG:
            print(f"[debug] processing line: {line}")
        return process_line_with_tools(
            line,
            debug=DEBUG,
            append_context=append_context,
            call_model=call_model,
            parse_response=parse_response,
            execute_tool_call=execute_tool_call_info,
        )

    def before_first_prompt() -> None:
        append_context({"role": "system", "content": INTERNAL_SYSTEM_PROMPT})
        user_prompt = load_user_prompt("AM_PROMPT_FILE", ".am_prompt", debug=DEBUG)
        if user_prompt:
            append_context({"role": "system", "content": user_prompt})

    return run_repl(
        initial_line=initial_line,
        before_first_prompt=before_first_prompt,
        process_line=process,
        after_each_prompt=state.reset_after_prompt,
    )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import os
import sys
from typing import Any

from agent_mode_cli.core.bash_tool import bash_command
from agent_mode_cli.core.agent_loop import ParseResult, ToolCallInfo, process_line_with_tools, safe_json_loads
from agent_mode_cli.core.cli_help import handle_common_flags
from agent_mode_cli.core.confirm import ConfirmState, prompt_for_confirmation, requires_confirmation
from agent_mode_cli.core.universal_context import ChatMessage, ToolCall, UniversalContext, to_openai_responses_input
from agent_mode_cli.core.openai_runtime import create_openai_client
from agent_mode_cli.core.prompt_file import load_user_prompt
from agent_mode_cli.core.repl import run_repl
from agent_mode_cli.core.system_prompt import build_internal_system_prompt
from agent_mode_cli.core.tool_specs import build_openai_tools
from agent_mode_cli.core.web_fetch import web_fetch
from agent_mode_cli.core.js_web_fetch import js_web_fetch


DEBUG = os.getenv("AM_DEBUG", "").lower() in ("1", "true", "yes", "on")
DEFAULT_MODEL = os.getenv("AM_MODEL", "gpt-5.1-codex")

INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("AM")
def main(argv: list[str] | None = None) -> int:
    global DEBUG, DEFAULT_MODEL

    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_mode_cli import __version__

    flag_exit = handle_common_flags(
        argv,
        usage="am <prompt...>",
        description="Agent-mode CLI backed by OpenAI.",
        env_lines=(
            "OPENAI_API_KEY  required",
            "AM_MODEL        optional (default: gpt-5.1-codex)",
            "AM_DEBUG        optional (1/true enables debug)",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit
    initial_line = " ".join(argv) if argv else None

    try:
        client = create_openai_client()
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    context = UniversalContext()
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
        "web_fetch": lambda url="", timeout_seconds=20, max_bytes=1500000, extract_text=True, max_chars=20000, headers=None: web_fetch(
            url,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
            extract_text=extract_text,
            max_chars=max_chars,
            headers=headers,
        ),
        "js_web_fetch": lambda url="", timeout_seconds=30, wait_until="networkidle", extract_text=True, max_chars=20000, user_agent=None: js_web_fetch(
            url,
            timeout_seconds=timeout_seconds,
            wait_until=wait_until,
            extract_text=extract_text,
            max_chars=max_chars,
            user_agent=user_agent,
        ),
    }

    def append_context(message: ChatMessage) -> None:
        context.append(message, debug=DEBUG)

    def call_model() -> Any:
        if DEBUG:
            print(f"[debug] calling model with context of {len(context)} messages")
        return client.responses.create(model=DEFAULT_MODEL, tools=tools, input=to_openai_responses_input(context.messages))

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

    def parse_response(response: Any) -> ParseResult:
        output = getattr(response, "output", []) or []
        tool_calls: list[ToolCallInfo] = []
        has_function_call = any(getattr(item, "type", None) == "function_call" for item in output)

        context_entries: list[ChatMessage] = []
        if output and getattr(output[0], "type", None) == "reasoning":
            reasoning_text = stringify_reasoning(output[0])
            if reasoning_text:
                context_entries.append(ChatMessage(role="assistant", content=f"[reasoning]\n{reasoning_text}"))
                if DEBUG:
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
            # Store tool calls in the assistant turn; adapter will expand them.
            context_entries.append(ChatMessage(role="assistant", content="", tool_calls=calls_for_assistant))

        final_text = getattr(response, "output_text", None)
        if not tool_calls and final_text:
            context_entries.append(ChatMessage(role="assistant", content=final_text))

        return ParseResult(tool_calls=tool_calls, context_entries=context_entries, final_text=final_text)

    def execute_tool_call_info(call: ToolCallInfo) -> tuple[list[ChatMessage], str | None]:
        if DEBUG:
            print(f"[debug] executing tool: {call.name}")
        if requires_confirmation(call.name):
            if not prompt_for_confirmation(call.name, call.arguments, state):
                cancel_message = f"Tool '{call.name}' execution cancelled by user."
                return (
                    [ChatMessage(role="tool", content="cancelled by user", tool_name=call.name, tool_call_id=call.call_id)],
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
            [ChatMessage(role="tool", content=output_text, tool_name=call.name, tool_call_id=call.call_id)],
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
        append_context(ChatMessage(role="system", content=INTERNAL_SYSTEM_PROMPT))
        user_prompt = load_user_prompt("AM_PROMPT_FILE", ".am_prompt", debug=DEBUG)
        if user_prompt:
            append_context(ChatMessage(role="system", content=user_prompt))

    return run_repl(
        initial_line=initial_line,
        before_first_prompt=before_first_prompt,
        process_line=process,
        after_each_prompt=state.reset_after_prompt,
    )


if __name__ == "__main__":
    raise SystemExit(main())

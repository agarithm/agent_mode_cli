from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import ollama
from ollama import Client, RequestError, ResponseError

from agent_mode_cli.core.bash_tool import bash_command
from agent_mode_cli.core.agent_loop import ParseResult, ToolCallInfo, process_line_with_tools
from agent_mode_cli.core.cli_help import handle_common_flags
from agent_mode_cli.core.confirm import ConfirmState, prompt_for_confirmation, requires_confirmation
from agent_mode_cli.core.universal_context import ChatMessage, ToolCall, UniversalContext, to_ollama_chat_messages
from agent_mode_cli.core.ollama_runtime import prepare_runtime
from agent_mode_cli.core.prompt_file import load_user_prompt
from agent_mode_cli.core.repl import run_repl
from agent_mode_cli.core.system_prompt import build_internal_system_prompt
from agent_mode_cli.core.tool_specs import build_ollama_tools
from agent_mode_cli.core.web_fetch import web_fetch
from agent_mode_cli.core.js_web_fetch import js_web_fetch


DEBUG = os.getenv("OM_DEBUG", "").lower() in ("1", "true", "yes", "on")
DEFAULT_MODEL = os.getenv("OM_MODEL", "gpt-oss")

INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("OM")

def normalize_arguments(arguments: Any) -> Dict[str, Any]:
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


def main(argv: list[str] | None = None) -> int:
    global DEBUG, DEFAULT_MODEL

    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_mode_cli import __version__

    flag_exit = handle_common_flags(
        argv,
        usage="om <prompt...>",
        description="Agent-mode CLI backed by Ollama.",
        env_lines=(
            "OM_MODEL        optional (default: gpt-oss)",
            "OM_DEBUG        optional (1/true enables debug)",
            "OLLAMA_HOST     optional (use remote Ollama)",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit
    initial_line = " ".join(argv) if argv else None

    try:
        prepare_runtime(debug=DEBUG, log_prefix="[om]")
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    client: Optional[Client] = ollama.Client()
    context = UniversalContext()
    tools = build_ollama_tools("OM")
    state = ConfirmState(approve_all=False, debug=DEBUG)

    def set_debug_command(enabled: bool) -> str:
        nonlocal state
        global DEBUG
        DEBUG = bool(enabled)
        os.environ["OM_DEBUG"] = "1" if DEBUG else "0"
        state.debug = DEBUG
        return f"OM_DEBUG {'enabled' if DEBUG else 'disabled'}."

    def set_model_command(model: str) -> str:
        global DEFAULT_MODEL
        model = (model or "").strip()
        if not model:
            return "error: model is required"
        DEFAULT_MODEL = model
        os.environ["OM_MODEL"] = model
        return f"OM_MODEL set to {model}."

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

    def debug_dump_context() -> None:
        if not DEBUG:
            return
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

    def call_model() -> ollama.ChatResponse:
        if client is None:
            raise RuntimeError("ollama client is not initialized")
        debug_dump_context()
        try:
            return client.chat(model=DEFAULT_MODEL, messages=to_ollama_chat_messages(context.messages), tools=tools)
        except (RequestError, ResponseError) as exc:
            raise RuntimeError(str(exc)) from exc

    def parse_response(response: ollama.ChatResponse) -> ParseResult:
        message = response.message
        if message is None:
            error_text = "error: empty response from model"
            return ParseResult(tool_calls=[], context_entries=[ChatMessage(role="assistant", content=error_text)], final_text=error_text)
        assistant_content = (message.content or "").strip()
        raw_tool_calls = list(message.tool_calls or [])

        tool_calls: list[ToolCallInfo] = []
        assistant_tool_calls: list[ToolCall] = []
        if raw_tool_calls:
            for idx, call in enumerate(raw_tool_calls):
                function = getattr(call, "function", None)
                name = getattr(function, "name", None)
                if not name:
                    continue
                arguments = normalize_arguments(getattr(function, "arguments", None))
                call_id = f"call_{idx}"  # Ollama tool calls do not expose a stable call_id
                tool_calls.append(ToolCallInfo(name=name, arguments=arguments, raw=None, call_id=call_id))
                assistant_tool_calls.append(ToolCall(name=name, arguments=arguments, call_id=call_id))

        assistant_entry = ChatMessage(
            role=message.role or "assistant",
            content=assistant_content,
            tool_calls=assistant_tool_calls or None,
        )

        final_text = assistant_content or "(no content)"
        return ParseResult(tool_calls=tool_calls, context_entries=[assistant_entry], final_text=final_text)

    def execute_tool_call_info(call: ToolCallInfo) -> Tuple[Sequence[ChatMessage], Optional[str]]:
        name = call.name
        arguments = call.arguments
        if requires_confirmation(name):
            if not prompt_for_confirmation(name, arguments, state):
                output_text = "cancelled by user"
                return ([ChatMessage(role="tool", content=output_text, tool_name=name, tool_call_id=call.call_id)], f"Tool '{name}' execution cancelled by user.")
        handler = TOOL_FUNCTIONS.get(name)
        if handler is None:
            output_text = f"error: unknown tool '{name}'"
        else:
            try:
                output_text = (handler(**arguments) or "").strip() or "(no output)"
            except TypeError as exc:
                output_text = f"error: invalid arguments - {exc}"
            except Exception as exc:
                output_text = f"error: {exc}"
        return ([ChatMessage(role="tool", content=output_text, tool_name=name, tool_call_id=call.call_id)], None)

    def process(line: str) -> str:
        if DEBUG:
            print(f"[debug] processing line: {line}")
        try:
            return process_line_with_tools(
                line,
                debug=DEBUG,
                append_context=append_context,
                call_model=call_model,
                parse_response=parse_response,
                execute_tool_call=execute_tool_call_info,
            )
        except RuntimeError as exc:
            error_message = f"error: {exc}"
            append_context(ChatMessage(role="assistant", content=error_message))
            return error_message

    def before_first_prompt() -> None:
        append_context(ChatMessage(role="system", content=INTERNAL_SYSTEM_PROMPT))
        user_prompt = load_user_prompt("OM_PROMPT_FILE", ".om_prompt", debug=DEBUG)
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

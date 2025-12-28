from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from agent_mode_cli.core.agent_loop import ToolCallInfo, process_line_with_tools
from agent_mode_cli.core.bash_tool import bash_command
from agent_mode_cli.core.confirm import ConfirmState, prompt_for_confirmation, requires_confirmation
from agent_mode_cli.core.js_web_fetch import js_web_fetch
from agent_mode_cli.core.prompt_file import load_user_prompt
from agent_mode_cli.core.repl import run_repl
from agent_mode_cli.core.universal_context import ChatMessage, UniversalContext
from agent_mode_cli.core.web_fetch import web_fetch
from agent_mode_cli.core.provider_adapter import ProviderAdapter


@dataclass(frozen=True)
class AgentRunnerConfig:
    agent_name: str
    env_prefix: str
    debug_env: str
    model_env: str
    initial_debug: bool
    initial_model: str
    prompt_file_env: str
    prompt_file_default: str
    internal_system_prompt: str
    catch_runtime_errors: bool = False


def run_agent_repl(
    *,
    adapter: ProviderAdapter,
    build_tools: Callable[[str], Sequence[Dict[str, Any]]],
    config: AgentRunnerConfig,
    initial_line: Optional[str],
) -> int:
    debug = bool(config.initial_debug)
    model = (config.initial_model or "").strip()

    context = UniversalContext()
    tools = list(build_tools(config.env_prefix))
    state = ConfirmState(approve_all=False, debug=debug)

    def set_debug_command(enabled: bool) -> str:
        nonlocal debug
        debug = bool(enabled)
        os.environ[config.debug_env] = "1" if debug else "0"
        state.debug = debug
        return f"{config.debug_env} {'enabled' if debug else 'disabled'}."

    def set_model_command(model_name: str) -> str:
        nonlocal model
        model_name = (model_name or "").strip()
        if not model_name:
            return "error: model is required"
        model = model_name
        os.environ[config.model_env] = model
        return f"{config.model_env} set to {model}."

    tool_functions: Dict[str, Callable[..., str]] = {
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
        context.append(message, debug=debug)

    def call_model() -> Any:
        active_model = (model or "").strip() or (config.initial_model or "").strip()
        return adapter.call_model(model=active_model, tools=tools, context=context, debug=debug)

    def parse_response(response: Any):
        return adapter.parse_response(response, debug=debug)

    def execute_tool_call(call: ToolCallInfo) -> Tuple[Sequence[ChatMessage], Optional[str]]:
        if debug:
            print(f"[debug] executing tool: {call.name}")

        if requires_confirmation(call.name):
            if not prompt_for_confirmation(call.name, call.arguments, state):
                cancel_message = f"Tool '{call.name}' execution cancelled by user."
                return (
                    [ChatMessage(role="tool", content="cancelled by user", tool_name=call.name, tool_call_id=call.call_id)],
                    cancel_message,
                )

        handler = tool_functions.get(call.name)
        if handler is None:
            output_text = f"error: unknown tool '{call.name}'"
        else:
            try:
                output_text = (handler(**call.arguments) or "").strip() or "(no output)"
            except TypeError as exc:
                output_text = f"error: invalid arguments - {exc}"
            except Exception as exc:
                output_text = f"error: {exc}"

        return ([ChatMessage(role="tool", content=output_text, tool_name=call.name, tool_call_id=call.call_id)], None)

    def process(line: str) -> str:
        if debug:
            print(f"[debug] processing line: {line}")

        if not config.catch_runtime_errors:
            return process_line_with_tools(
                line,
                debug=debug,
                append_context=append_context,
                call_model=call_model,
                parse_response=parse_response,
                execute_tool_call=execute_tool_call,
            )

        try:
            return process_line_with_tools(
                line,
                debug=debug,
                append_context=append_context,
                call_model=call_model,
                parse_response=parse_response,
                execute_tool_call=execute_tool_call,
            )
        except RuntimeError as exc:
            error_message = f"error: {exc}"
            append_context(ChatMessage(role="assistant", content=error_message))
            return error_message

    def before_first_prompt() -> None:
        append_context(ChatMessage(role="system", content=config.internal_system_prompt))
        user_prompt = load_user_prompt(config.prompt_file_env, config.prompt_file_default, debug=debug)
        if user_prompt:
            append_context(ChatMessage(role="system", content=user_prompt))

    return run_repl(
        initial_line=initial_line,
        before_first_prompt=before_first_prompt,
        process_line=process,
        after_each_prompt=state.reset_after_prompt,
    )

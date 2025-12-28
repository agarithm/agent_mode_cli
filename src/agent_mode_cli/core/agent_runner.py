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


@dataclass(frozen=True)
class ProviderEntry:
    """Descriptor for an LLM provider used by a multi-provider REPL."""

    name: str
    description: str
    default_model: str
    build_tools: Callable[[str], Sequence[Dict[str, Any]]]
    create_adapter: Callable[[], ProviderAdapter]
    prepare_runtime: Optional[Callable[[bool], None]] = None


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


def run_agent_repl_multi_provider(
    *,
    providers: Mapping[str, ProviderEntry],
    initial_provider: str,
    config: AgentRunnerConfig,
    initial_line: Optional[str],
) -> int:
    """Run a REPL that can switch providers mid-session.

    Commands intercepted locally (not sent to the model):
    - providers | list providers
    - use <name> | use provider <name>
    """

    if not providers:
        raise ValueError("providers mapping is required")

    provider_key = (initial_provider or "").strip().lower()
    if provider_key not in providers:
        available = ", ".join(sorted(providers.keys()))
        raise ValueError(f"unknown provider '{provider_key}'. Available: {available}")

    debug = bool(config.initial_debug)
    # Model is tracked per-provider so switching doesn't inherit a nonsense model name.
    models_by_provider: Dict[str, str] = {
        key: (entry.default_model or "").strip() for key, entry in providers.items()
    }
    initial_model_env = (os.getenv(config.model_env) or "").strip()
    if initial_model_env:
        models_by_provider[provider_key] = initial_model_env
    elif (config.initial_model or "").strip():
        models_by_provider[provider_key] = (config.initial_model or "").strip()

    context = UniversalContext()
    state = ConfirmState(approve_all=False, debug=debug)

    adapter_cache: Dict[str, ProviderAdapter] = {}

    active_provider = provider_key
    active_adapter: ProviderAdapter
    active_tools: Sequence[Dict[str, Any]]

    def _get_or_create_adapter(name: str) -> ProviderAdapter:
        if name in adapter_cache:
            return adapter_cache[name]
        entry = providers[name]
        if entry.prepare_runtime is not None:
            entry.prepare_runtime(debug)
        adapter_cache[name] = entry.create_adapter()
        return adapter_cache[name]

    def _apply_active_provider(name: str) -> None:
        nonlocal active_provider, active_adapter, active_tools
        name = (name or "").strip().lower()
        if name not in providers:
            available = ", ".join(sorted(providers.keys()))
            raise RuntimeError(f"unknown provider '{name}'. Available: {available}")
        active_provider = name
        active_adapter = _get_or_create_adapter(name)
        active_tools = list(providers[name].build_tools(config.env_prefix))

        active_model = (models_by_provider.get(active_provider) or "").strip()
        if active_model:
            os.environ[config.model_env] = active_model

    _apply_active_provider(active_provider)

    def set_debug_command(enabled: bool) -> str:
        nonlocal debug
        debug = bool(enabled)
        os.environ[config.debug_env] = "1" if debug else "0"
        state.debug = debug
        return f"{config.debug_env} {'enabled' if debug else 'disabled'}."

    def set_model_command(model: str) -> str:
        model = (model or "").strip()
        if not model:
            return "error: model is required"
        models_by_provider[active_provider] = model
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
        active_model = (models_by_provider.get(active_provider) or "").strip()
        if not active_model:
            active_model = (providers[active_provider].default_model or "").strip()
        return active_adapter.call_model(model=active_model, tools=active_tools, context=context, debug=debug)

    def parse_response(response: Any):
        return active_adapter.parse_response(response, debug=debug)

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

    def _format_providers() -> str:
        lines: list[str] = ["Providers:"]
        for key in sorted(providers.keys()):
            entry = providers[key]
            current = " (current)" if key == active_provider else ""
            model = (models_by_provider.get(key) or entry.default_model or "").strip()
            model_part = f" model={model}" if model else ""
            lines.append(f"- {key}{current}: {entry.description}{model_part}")
        return "\n".join(lines)

    def _try_handle_local_command(line: str) -> Optional[str]:
        raw = (line or "").strip()
        if not raw:
            return None
        lowered = raw.lower().strip()

        if lowered in {"providers", "list providers", ":providers", ":list providers"}:
            return _format_providers()

        for prefix in ("use provider ", "use ", ":use ", ":provider ", "provider "):
            if lowered.startswith(prefix):
                target = raw[len(prefix) :].strip()
                if not target:
                    return "error: provider name is required"
                try:
                    _apply_active_provider(target)
                except Exception as exc:
                    return f"error: {exc}"
                # Tell the model too, so it knows future tool schemas may differ.
                append_context(ChatMessage(role="system", content=f"Provider switched to {active_provider}."))
                return f"Using provider: {active_provider}"
        return None

    def process(line: str) -> str:
        if debug:
            print(f"[debug] processing line: {line}")

        local_result = _try_handle_local_command(line)
        if local_result is not None:
            return local_result

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
        append_context(ChatMessage(role="system", content=f"Provider switched to {active_provider}."))

    return run_repl(
        initial_line=initial_line,
        before_first_prompt=before_first_prompt,
        process_line=process,
        after_each_prompt=state.reset_after_prompt,
    )

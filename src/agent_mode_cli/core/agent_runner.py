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
from agent_mode_cli.providers.base import ProviderAdapter
from agent_mode_cli.core.runtime_settings import RuntimeSettings


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
    max_tool_iterations: int = 100
    max_tool_seconds: Optional[float] = None


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

    settings = RuntimeSettings(
        debug=bool(config.initial_debug),
        max_tool_iterations=int(config.max_tool_iterations),
        max_tool_seconds=config.max_tool_seconds,
    )
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
    state = ConfirmState(approve_all=False, debug=settings.debug)

    adapter_cache: Dict[str, ProviderAdapter] = {}

    active_provider = provider_key
    active_adapter: ProviderAdapter
    active_tools: Sequence[Dict[str, Any]]

    def _get_or_create_adapter(name: str) -> ProviderAdapter:
        if name in adapter_cache:
            return adapter_cache[name]
        entry = providers[name]
        if entry.prepare_runtime is not None:
            entry.prepare_runtime(settings.debug)
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

    tool_functions: Dict[str, Callable[..., str]] = {
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
        context.append(message, debug=settings.debug)

    def call_model() -> Any:
        active_model = (models_by_provider.get(active_provider) or "").strip()
        if not active_model:
            active_model = (providers[active_provider].default_model or "").strip()
        return active_adapter.call_model(model=active_model, tools=active_tools, context=context, debug=settings.debug)

    def parse_response(response: Any):
        return active_adapter.parse_response(response, debug=settings.debug)

    def execute_tool_call(call: ToolCallInfo) -> Tuple[Sequence[ChatMessage], Optional[str]]:
        if settings.debug:
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

    def _format_settings() -> str:
        lines: list[str] = [settings.format()]
        model = (models_by_provider.get(active_provider) or providers[active_provider].default_model or "").strip()
        lines.append(f"- provider: {active_provider}")
        if model:
            lines.append(f"- model: {model}")
        return "\n".join(lines)

    def _help_text() -> str:
        return "\n".join(
            [
                "Commands:",
                "- help | :help                 Show this help.",
                "- providers | :providers        List available providers.",
                "- use <name> | :use <name>      Switch provider for this session.",
                "- settings | :settings          Show current settings.",
                "- debug <on|off>                Toggle debug logging.",
                "- model <name>                  Set model for the current provider.",
                "- max_tool_iterations <n>        Max tool-loop iterations per prompt.",
                "- max_tool_seconds <sec|off>     Max wall-clock seconds in tool loop.",
                "- quit | q | exit                Exit the REPL.",
                "",
                "Notes:",
                "- Model is tracked per provider; switching providers keeps history.",
                "- Settings are per-process; they reset when you restart.",
            ]
        )

    def _try_handle_local_command(line: str) -> Optional[str]:
        raw = (line or "").strip()
        if not raw:
            return None
        lowered = raw.lower().strip()

        if lowered in {"help", ":help", "?", ":?", "commands", ":commands"}:
            return _help_text()

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

        if lowered in {"settings", ":settings", "limits", ":limits"}:
            return _format_settings()

        for prefix in ("debug ", ":debug "):
            if lowered.startswith(prefix):
                result = settings.set_debug_from_text(raw[len(prefix) :])
                os.environ[config.debug_env] = "1" if settings.debug else "0"
                state.debug = settings.debug
                return result

        for prefix in ("model ", ":model "):
            if lowered.startswith(prefix):
                new_model = raw[len(prefix) :].strip()
                if not new_model:
                    return "error: model name is required"
                models_by_provider[active_provider] = new_model
                os.environ[config.model_env] = new_model
                return f"{config.model_env} set to {new_model}."

        for prefix in ("max_tool_iterations ", ":max_tool_iterations "):
            if lowered.startswith(prefix):
                return settings.set_max_tool_iterations_from_text(raw[len(prefix) :])

        for prefix in ("max_tool_seconds ", ":max_tool_seconds "):
            if lowered.startswith(prefix):
                return settings.set_max_tool_seconds_from_text(raw[len(prefix) :])
        return None

    def process(line: str) -> str:
        if settings.debug:
            print(f"[debug] processing line: {line}")

        local_result = _try_handle_local_command(line)
        if local_result is not None:
            return local_result

        if not config.catch_runtime_errors:
            return process_line_with_tools(
                line,
                debug=settings.debug,
                append_context=append_context,
                call_model=call_model,
                parse_response=parse_response,
                execute_tool_call=execute_tool_call,
                max_tool_iterations=settings.max_tool_iterations,
                max_tool_seconds=settings.max_tool_seconds,
            )

        try:
            return process_line_with_tools(
                line,
                debug=settings.debug,
                append_context=append_context,
                call_model=call_model,
                parse_response=parse_response,
                execute_tool_call=execute_tool_call,
                max_tool_iterations=settings.max_tool_iterations,
                max_tool_seconds=settings.max_tool_seconds,
            )
        except RuntimeError as exc:
            error_message = f"error: {exc}"
            append_context(ChatMessage(role="assistant", content=error_message))
            return error_message

    def before_first_prompt() -> None:
        append_context(ChatMessage(role="system", content=config.internal_system_prompt))
        user_prompt = load_user_prompt(config.prompt_file_env, config.prompt_file_default, debug=settings.debug)
        if user_prompt:
            append_context(ChatMessage(role="system", content=user_prompt))
        append_context(ChatMessage(role="system", content=f"Provider switched to {active_provider}."))

    return run_repl(
        initial_line=initial_line,
        before_first_prompt=before_first_prompt,
        process_line=process,
        after_each_prompt=state.reset_after_prompt,
    )

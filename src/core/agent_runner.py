from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from core.agent_loop import ToolCallInfo, process_line_with_tools
from core.confirm import ConfirmState, prompt_for_confirmation, requires_confirmation
from tools import bash_command
from tools import edit_file
from tools import http_fetch
from tools import list_dir, read_file
from tools import python_exec
from tools import search_files
from core.prompt_file import load_user_prompt
from core.repl import run_repl
from core.universal_context import ChatMessage, UniversalContext
from providers.base import ProviderAdapter, ProviderRateLimitError
from core.runtime_settings import RuntimeSettings


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
    validate_model: Optional[Callable[[str, bool], str]] = None
    fallback_providers: Sequence[str] = ()


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
    preferred_provider = provider_key
    active_adapter: ProviderAdapter
    active_tools: Sequence[Dict[str, Any]]

    rate_limited_until: Dict[str, float] = {}
    default_retry_after_seconds = float(os.getenv("AI_FALLBACK_DEFAULT_RETRY_AFTER", "300") or "300")

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

        # Validate model if provider has validation callback
        entry = providers[name]
        if entry.validate_model is not None:
            current_model = models_by_provider.get(name) or entry.default_model
            validated_model = entry.validate_model(current_model, settings.debug)
            models_by_provider[name] = validated_model

        active_provider = name
        active_adapter = _get_or_create_adapter(name)
        active_tools = list(providers[name].build_tools(config.env_prefix))

        active_model = (models_by_provider.get(active_provider) or "").strip()
        if active_model:
            os.environ[config.model_env] = active_model

    def _is_rate_limited(name: str) -> bool:
        until = rate_limited_until.get(name)
        if not until:
            return False
        return time.monotonic() < until

    def _remaining_rate_limit_seconds(name: str) -> Optional[float]:
        until = rate_limited_until.get(name)
        if not until:
            return None
        remaining = until - time.monotonic()
        return remaining if remaining > 0 else 0.0

    def _maybe_resume_preferred_provider() -> None:
        nonlocal active_provider
        if active_provider == preferred_provider:
            return
        if _is_rate_limited(preferred_provider):
            return
        _apply_active_provider(preferred_provider)
        _announce_active_provider_and_model("preferred provider available again")

    _apply_active_provider(active_provider)

    tool_functions: Dict[str, Callable[..., str]] = {
        "list_dir": lambda path=".", recursive=False, max_depth=2, max_entries=2000, include_metadata=False: list_dir(
            path,
            recursive=recursive,
            max_depth=max_depth,
            max_entries=max_entries,
            include_metadata=include_metadata,
        ),
        "read_file": lambda path, offset=0, length=None: read_file(
            path,
            offset=offset,
            length=length,
        ),
        "python_exec": lambda code="", input=None, timeout_seconds=10, max_chars=20000: python_exec(
            code,
            input=input,
            timeout_seconds=timeout_seconds,
            max_chars=max_chars,
        ),
        "search_files": lambda query, paths=None, is_regex=True, ignore_case=False, glob=None, context_lines=0, max_matches=200, max_chars=40000: search_files(
            query,
            paths=paths,
            is_regex=is_regex,
            ignore_case=ignore_case,
            glob=glob,
            context_lines=context_lines,
            max_matches=max_matches,
            max_chars=max_chars,
        ),
        "edit_file": lambda path, edits=None, mode=None, content=None, dry_run=False, make_backup=True: edit_file(
            path,
            edits,
            mode=mode,
            content=content,
            dry_run=dry_run,
            make_backup=make_backup,
        ),
        "bash": lambda command="": bash_command(command),
        "http_fetch": lambda url="", mode="simple", timeout_seconds=None, max_bytes=1500000, extract_text=True, max_chars=20000, headers=None, wait_until="networkidle", user_agent=None: http_fetch(
            url,
            mode=mode,
            timeout_seconds=timeout_seconds,
            max_bytes=max_bytes,
            extract_text=extract_text,
            max_chars=max_chars,
            headers=headers,
            wait_until=wait_until,
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

    def _active_model_for(name: str) -> str:
        model = (models_by_provider.get(name) or "").strip()
        if not model:
            model = (providers[name].default_model or "").strip()
        return model

    def _announce_active_provider_and_model(reason: str) -> None:
        model = _active_model_for(active_provider)
        suffix = f" ({reason})" if reason else ""
        append_context(
            ChatMessage(
                role="system",
                content=(
                    f"Active provider: {active_provider}. Active model: {model}.{suffix} "
                    "If asked which model you are, answer with the active model identifier above."
                ),
            )
        )

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
                "- :provider [name]              Show or switch provider.",
                "- :model [id]                   Show or switch model for current provider.",
                "- :settings [subcommand...]      Show or change runtime settings.",
                "- quit | q | exit                Exit the REPL.",
                "",
                "Notes:",
                "- Model is tracked per provider; switching providers keeps history.",
                "- Settings are per-process; they reset when you restart.",
                "",
                "Settings subcommands:",
                "- :settings debug <on|off>",
                "- :settings provider <name>",
                "- :settings model <id>",
                "- :settings max_tool_iterations <n>",
                "- :settings max_tool_seconds <sec|off>",
            ]
        )

    def _try_handle_local_command(line: str) -> Optional[str]:
        nonlocal preferred_provider
        raw = (line or "").strip()
        if not raw:
            return None
        lowered = raw.lower().strip()

        if lowered in {"help", ":help", "?", ":?", "commands", ":commands"}:
            return f"{_help_text()}\n\n{_format_settings()}"

        def _format_provider_status() -> str:
            lines: list[str] = [f"Current provider: {active_provider}", "", _format_providers()]
            return "\n".join(lines)

        def _format_model_status(*, provider_name: str) -> str:
            current_model = _active_model_for(provider_name)
            try:
                adapter = _get_or_create_adapter(provider_name)
                models = list(adapter.list_models(debug=settings.debug))
            except Exception as exc:
                return f"Current model: {current_model}\n\nerror: failed to list models for '{provider_name}': {exc}"

            models = [m for m in models if (m or "").strip()]
            if not models:
                return f"Current model: {current_model}\n\nAvailable models: (none found)"

            max_lines = int(os.getenv("AI_MODELS_LIST_MAX", "200") or "200")
            lines: list[str] = [f"Current model: {current_model}", f"Available models for {provider_name} ({len(models)}):"]
            for mid in models[:max_lines]:
                prefix = "*" if mid == current_model else "-"
                lines.append(f"{prefix} {mid}")
            if len(models) > max_lines:
                lines.append(f"... ({len(models) - max_lines} more; set AI_MODELS_LIST_MAX to increase)")
            return "\n".join(lines)

        # Back-compat aliases (not advertised).
        if lowered in {":providers", ":list providers"}:
            lowered = ":provider"
            raw = ":provider"
        if lowered.startswith(":use "):
            lowered = ":provider " + lowered[len(":use ") :]
            raw = ":provider " + raw[len(":use ") :]

        if lowered == ":provider" or lowered.startswith(":provider "):
            target = raw[len(":provider") :].strip()
            if not target:
                return _format_provider_status()
            provider_name = target.strip().lower()
            try:
                _apply_active_provider(provider_name)
            except Exception as exc:
                return f"error: {exc}"
            preferred_provider = active_provider
            _announce_active_provider_and_model("user requested")
            return _format_provider_status()

        if lowered == ":model" or lowered.startswith(":model "):
            target = raw[len(":model") :].strip()
            if not target:
                return _format_model_status(provider_name=active_provider)
            new_model = target
            models_by_provider[active_provider] = new_model
            os.environ[config.model_env] = new_model
            _announce_active_provider_and_model("user requested")
            return _format_model_status(provider_name=active_provider)

        if lowered == ":settings" or lowered in {":limits"}:
            return _format_settings()

        if lowered.startswith(":settings "):
            rest = raw[len(":settings ") :].strip()
            if not rest:
                return _format_settings()
            parts = rest.split()
            action = (parts[0] or "").strip().lower()
            value = " ".join(parts[1:]).strip()

            if action == "debug":
                result = settings.set_debug_from_text(value)
                os.environ[config.debug_env] = "1" if settings.debug else "0"
                state.debug = settings.debug
                return f"{result}\n\n{_format_settings()}"

            if action == "provider":
                if not value:
                    return "error: provider name is required"
                try:
                    _apply_active_provider(value.strip().lower())
                except Exception as exc:
                    return f"error: {exc}"
                preferred_provider = active_provider
                _announce_active_provider_and_model("user requested")
                return _format_settings()

            if action == "model":
                if not value:
                    return "error: model id is required"
                models_by_provider[active_provider] = value
                os.environ[config.model_env] = value
                _announce_active_provider_and_model("user requested")
                return _format_settings()

            if action == "max_tool_iterations":
                result = settings.set_max_tool_iterations_from_text(value)
                return f"{result}\n\n{_format_settings()}"

            if action == "max_tool_seconds":
                result = settings.set_max_tool_seconds_from_text(value)
                return f"{result}\n\n{_format_settings()}"

            return (
                "error: unknown settings subcommand. Use one of: "
                "debug, provider, model, max_tool_iterations, max_tool_seconds"
            )
        return None

    def process(line: str) -> str:
        if settings.debug:
            print(f"[debug] processing line: {line}")

        _maybe_resume_preferred_provider()

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

        suppress_next_user_append = False

        def append_context_maybe_suppress(message: ChatMessage) -> None:
            nonlocal suppress_next_user_append
            if suppress_next_user_append and message.role == "user" and message.content == line:
                suppress_next_user_append = False
                return
            append_context(message)

        while True:
            try:
                return process_line_with_tools(
                    line,
                    debug=settings.debug,
                    append_context=append_context_maybe_suppress,
                    call_model=call_model,
                    parse_response=parse_response,
                    execute_tool_call=execute_tool_call,
                    max_tool_iterations=settings.max_tool_iterations,
                    max_tool_seconds=settings.max_tool_seconds,
                )

            except ProviderRateLimitError as exc:
                if _handle_provider_rate_limit(exc):
                    suppress_next_user_append = True
                    continue
                remaining = _remaining_rate_limit_seconds(preferred_provider)
                message = f"error: rate limit on {exc.provider}"
                if remaining:
                    message += f" (preferred provider retry in ~{remaining:.1f}s)"
                append_context(ChatMessage(role="assistant", content=message))
                return message
            except RuntimeError as exc:
                error_message = f"error: {exc}"
                append_context(ChatMessage(role="assistant", content=error_message))
                return error_message

    def _handle_provider_rate_limit(error: ProviderRateLimitError) -> bool:
        """Handle a provider 429 by switching providers.

        Returns True if a new provider was selected and the caller should retry.
        Returns False if no alternatives are available.
        """

        nonlocal active_provider

        retry_after = error.retry_after
        if retry_after is None or retry_after <= 0:
            retry_after = default_retry_after_seconds

        rate_limited_until[active_provider] = time.monotonic() + float(retry_after)

        current_entry = providers[active_provider]
        fallbacks = list(current_entry.fallback_providers or ())
        if not fallbacks:
            return False

        for candidate in fallbacks:
            candidate = (candidate or "").strip().lower()
            if not candidate or candidate == active_provider:
                continue
            if candidate not in providers:
                continue
            if _is_rate_limited(candidate):
                continue

            _apply_active_provider(candidate)
            _announce_active_provider_and_model(f"failover from rate-limited provider '{error.provider}'")
            return True

        return False


    def before_first_prompt() -> None:
        append_context(ChatMessage(role="system", content=config.internal_system_prompt))
        user_prompt = load_user_prompt(config.prompt_file_env, config.prompt_file_default, debug=settings.debug)
        if user_prompt:
            append_context(ChatMessage(role="system", content=user_prompt))
        _announce_active_provider_and_model("session start")

    return run_repl(
        initial_line=initial_line,
        before_first_prompt=before_first_prompt,
        process_line=process,
        after_each_prompt=state.reset_after_prompt,
    )

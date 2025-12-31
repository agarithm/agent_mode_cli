from __future__ import annotations

import os
import sys
from typing import Optional

import ollama

from core.agent_runner import AgentRunnerConfig, ProviderEntry, run_agent_repl
from core.cli_help import handle_common_flags
from core.system_prompt import build_internal_system_prompt
from providers.ollama.adapter import OllamaProviderAdapter
from providers.ollama.runtime import prepare_runtime
from providers.ollama.tools import build_tools as build_ollama_tools
from providers.ollama.validation import ensure_ollama_model
from providers.copilot.adapter import CopilotProviderAdapter
from providers.copilot.runtime import create_github_models_client
from providers.copilot.tools import build_tools as build_copilot_tools
from providers.openai.adapter import OpenAIProviderAdapter
from providers.openai.runtime import create_openai_client
from providers.openai.tools import build_tools as build_openai_tools


INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("AI")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from version import __version__

    debug = os.getenv("AI_DEBUG", "").lower() in ("1", "true", "yes", "on")
    env_provider = os.getenv("AI_PROVIDER", "ollama").strip().lower() or "ollama"
    base_ollama_model = os.getenv("AI_MODEL", "gpt-oss:latest")

    flag_exit = handle_common_flags(
        argv,
        usage="ai [provider] [model] [prompt...]",
        description=(
            "Agent-mode CLI (full REPL) with dynamic provider switching. "
            "Default provider is Ollama (local)."
        ),
        env_lines=(
            "AI_PROVIDER     optional (default: ollama)",
            "AI_MODEL        optional (default: qwen2.5-coder:32b; applies to current provider)",
            "AI_DEBUG        optional (1/true enables debug)",
            "AI_PROMPT_FILE  optional (default: ~/.ai_prompt)",
            "OPENAI_API_KEY  required for OpenAI provider",
            "GITHUB_TOKEN    required for Copilot provider (GitHub Models)",
            "OLLAMA_HOST     optional (use remote Ollama)",
            "First CLI args  optional magic keywords for provider/model",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit

    def _create_ollama_adapter() -> OllamaProviderAdapter:
        client = ollama.Client()
        return OllamaProviderAdapter(client=client)

    def _create_openai_adapter() -> OpenAIProviderAdapter:
        client = create_openai_client()
        return OpenAIProviderAdapter(client=client)

    def _create_copilot_adapter() -> CopilotProviderAdapter:
        client = create_github_models_client()
        return CopilotProviderAdapter(client=client)

    def _build_providers(default_ollama_model: str, overrides: dict[str, str] | None = None) -> dict[str, ProviderEntry]:
        overrides = overrides or {}
        return {
            "ollama": ProviderEntry(
                name="ollama",
                description="Local Ollama (default)",
                default_model=overrides.get("ollama") or default_ollama_model,
                build_tools=build_ollama_tools,
                create_adapter=_create_ollama_adapter,
                prepare_runtime=lambda debug: prepare_runtime(debug=debug, log_prefix="[ai]"),
                validate_model=lambda model, debug: ensure_ollama_model(model, debug=debug, log_prefix="[ai]"),
            ),
            "copilot": ProviderEntry(
                name="copilot",
                description="GitHub Copilot / GitHub Models (requires GITHUB_TOKEN)",
                default_model=overrides.get("copilot") or os.getenv("AI_COPILOT_MODEL", "xai/grok-3"),
                build_tools=build_copilot_tools,
                create_adapter=_create_copilot_adapter,
                prepare_runtime=None,
                fallback_providers=(
                    os.getenv("AI_COPILOT_FALLBACK_PRIMARY", "openai"),
                    os.getenv("AI_COPILOT_FALLBACK_SECONDARY", "ollama"),
                ),
            ),
            "openai": ProviderEntry(
                name="openai",
                description="OpenAI (requires OPENAI_API_KEY)",
                default_model=overrides.get("openai") or os.getenv("AI_OPENAI_MODEL", "gpt-5.1-codex"),
                build_tools=build_openai_tools,
                create_adapter=_create_openai_adapter,
                prepare_runtime=None,
                fallback_providers=(
                    os.getenv("AI_OPENAI_FALLBACK_PRIMARY", "copilot"),
                    os.getenv("AI_OPENAI_FALLBACK_SECONDARY", "ollama"),
                ),
            ),
        }

    cli_prevalidated_models: dict[str, str] = {}
    cli_adapter_cache: dict[str, object] = {}

    def _extract_cli_overrides(raw_args: list[str], providers_map: dict[str, ProviderEntry]) -> tuple[Optional[str], Optional[str], list[str]]:
        remaining = list(raw_args)
        if not remaining:
            return None, None, []

        candidate_provider = (remaining[0] or "").strip().lower()
        if candidate_provider not in providers_map:
            return None, None, remaining

        provider_override = candidate_provider
        tokens_after_provider = remaining[1:]

        if not tokens_after_provider:
            return provider_override, None, []

        candidate_model = tokens_after_provider[0]

        def _candidate_is_valid_model(provider_name: str, candidate: str) -> tuple[bool, Optional[str]]:
            entry = providers_map[provider_name]
            normalized = (candidate or "").strip()
            if not normalized:
                return False, None
            if entry.validate_model is not None:
                try:
                    validated = entry.validate_model(normalized, debug)
                    cli_prevalidated_models[provider_name] = validated
                    return True, validated
                except Exception as exc:
                    if debug:
                        print(
                            f"[debug] CLI model '{normalized}' rejected for provider '{provider_name}': {exc}",
                            file=sys.stderr,
                        )
                    return False, None

            adapter = cli_adapter_cache.get(provider_name)
            if adapter is None:
                try:
                    if entry.prepare_runtime is not None:
                        entry.prepare_runtime(debug)
                    adapter = entry.create_adapter()
                    cli_adapter_cache[provider_name] = adapter
                except Exception as exc:
                    if debug:
                        print(
                            f"[debug] CLI could not initialize adapter for provider '{provider_name}': {exc}",
                            file=sys.stderr,
                        )
                    return False, None
            try:
                available = [m.strip() for m in adapter.list_models(debug=debug)]  # type: ignore[attr-defined]
            except Exception as exc:
                if debug:
                    print(
                        f"[debug] CLI could not list models for provider '{provider_name}': {exc}",
                        file=sys.stderr,
                    )
                return False, None
            available = [m for m in available if m]
            if normalized in available:
                return True, normalized
            if debug:
                print(
                    f"[debug] CLI candidate '{normalized}' not found in provider '{provider_name}' catalog",
                    file=sys.stderr,
                )
            return False, None

        is_valid_model, normalized_model = _candidate_is_valid_model(provider_override, candidate_model)
        if is_valid_model and normalized_model:
            return provider_override, normalized_model, tokens_after_provider[1:]
        return provider_override, None, tokens_after_provider

    providers = _build_providers(base_ollama_model)
    provider_override, model_override, prompt_tokens = _extract_cli_overrides(argv, providers)

    provider_model_overrides: dict[str, str] = {}
    if provider_override and model_override:
        provider_model_overrides[provider_override] = model_override

    if provider_override:
        os.environ["AI_PROVIDER"] = provider_override
    if provider_override and model_override:
        os.environ["AI_MODEL"] = model_override

    initial_provider = provider_override or env_provider
    default_ollama_model = cli_prevalidated_models.get("ollama") or base_ollama_model

    if initial_provider == "ollama" and "ollama" not in cli_prevalidated_models:
        try:
            ensured_model = ensure_ollama_model(
                provider_model_overrides.get("ollama") or default_ollama_model,
                debug=debug,
                log_prefix="[ai]",
            )
            default_ollama_model = ensured_model
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    providers = _build_providers(default_ollama_model, overrides=provider_model_overrides)
    initial_line = " ".join(prompt_tokens) if prompt_tokens else None

    runner_config = AgentRunnerConfig(
        agent_name="AI",
        env_prefix="AI",
        debug_env="AI_DEBUG",
        model_env="AI_MODEL",
        initial_debug=debug,
        initial_model=default_ollama_model,
        prompt_file_env="AI_PROMPT_FILE",
        prompt_file_default=".ai_prompt",
        internal_system_prompt=INTERNAL_SYSTEM_PROMPT,
        catch_runtime_errors=True,
    )

    try:
        return run_agent_repl(
            providers=providers,
            initial_provider=initial_provider,
            config=runner_config,
            initial_line=initial_line,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

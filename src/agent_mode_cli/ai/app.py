from __future__ import annotations

import os
import sys

import ollama

from agent_mode_cli.core.agent_runner import AgentRunnerConfig, ProviderEntry, run_agent_repl
from agent_mode_cli.core.cli_help import handle_common_flags
from agent_mode_cli.core.system_prompt import build_internal_system_prompt
from agent_mode_cli.providers.ollama.adapter import OllamaProviderAdapter
from agent_mode_cli.providers.ollama.runtime import prepare_runtime
from agent_mode_cli.providers.ollama.tools import build_tools as build_ollama_tools
from agent_mode_cli.providers.copilot.adapter import CopilotProviderAdapter
from agent_mode_cli.providers.copilot.runtime import create_github_models_client
from agent_mode_cli.providers.copilot.tools import build_tools as build_copilot_tools
from agent_mode_cli.providers.openai.adapter import OpenAIProviderAdapter
from agent_mode_cli.providers.openai.runtime import create_openai_client
from agent_mode_cli.providers.openai.tools import build_tools as build_openai_tools


INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("AI")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_mode_cli import __version__

    debug = os.getenv("AI_DEBUG", "").lower() in ("1", "true", "yes", "on")
    default_provider = os.getenv("AI_PROVIDER", "ollama").strip().lower() or "ollama"
    default_ollama_model = os.getenv("AI_MODEL", "gpt-oss")

    flag_exit = handle_common_flags(
        argv,
        usage="ai [prompt...]",
        description=(
            "Agent-mode CLI (full REPL) with dynamic provider switching. "
            "Default provider is Ollama (local)."
        ),
        env_lines=(
            "AI_PROVIDER     optional (default: ollama)",
            "AI_MODEL        optional (default: gpt-oss; applies to current provider)",
            "AI_DEBUG        optional (1/true enables debug)",
            "AI_PROMPT_FILE  optional (default: ~/.ai_prompt)",
            "OPENAI_API_KEY  required for OpenAI provider",
            "GITHUB_TOKEN    required for Copilot provider (GitHub Models)",
            "OLLAMA_HOST     optional (use remote Ollama)",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit

    initial_line = " ".join(argv) if argv else None

    def _create_ollama_adapter() -> OllamaProviderAdapter:
        client = ollama.Client()
        return OllamaProviderAdapter(client=client)

    def _create_openai_adapter() -> OpenAIProviderAdapter:
        client = create_openai_client()
        return OpenAIProviderAdapter(client=client)

    def _create_copilot_adapter() -> CopilotProviderAdapter:
        client = create_github_models_client()
        return CopilotProviderAdapter(client=client)

    providers = {
        "ollama": ProviderEntry(
            name="ollama",
            description="Local Ollama (default)",
            default_model=default_ollama_model,
            build_tools=build_ollama_tools,
            create_adapter=_create_ollama_adapter,
            prepare_runtime=lambda debug: prepare_runtime(debug=debug, log_prefix="[ai]"),
        ),
        "copilot": ProviderEntry(
            name="copilot",
            description="GitHub Copilot / GitHub Models (requires GITHUB_TOKEN)",
            default_model=os.getenv("AI_COPILOT_MODEL", "google/gemini-latest"),
            build_tools=build_copilot_tools,
            create_adapter=_create_copilot_adapter,
            prepare_runtime=None,
        ),
        "openai": ProviderEntry(
            name="openai",
            description="OpenAI (requires OPENAI_API_KEY)",
            default_model=os.getenv("AI_OPENAI_MODEL", "gpt-5.1-codex"),
            build_tools=build_openai_tools,
            create_adapter=_create_openai_adapter,
            prepare_runtime=None,
        ),
    }

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
            initial_provider=default_provider,
            config=runner_config,
            initial_line=initial_line,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

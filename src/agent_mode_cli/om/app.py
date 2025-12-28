from __future__ import annotations

import os
import sys
from typing import Optional

import ollama
from ollama import Client

from agent_mode_cli.core.agent_runner import AgentRunnerConfig, run_agent_repl
from agent_mode_cli.core.cli_help import handle_common_flags
from agent_mode_cli.core.ollama_runtime import prepare_runtime
from agent_mode_cli.core.ollama_adapter import OllamaProviderAdapter
from agent_mode_cli.core.system_prompt import build_internal_system_prompt
from agent_mode_cli.core.tool_specs import build_ollama_tools


DEBUG = os.getenv("OM_DEBUG", "").lower() in ("1", "true", "yes", "on")
DEFAULT_MODEL = os.getenv("OM_MODEL", "gpt-oss")

INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("OM")


def main(argv: list[str] | None = None) -> int:
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
    if client is None:
        print("error: ollama client is not initialized", file=sys.stderr)
        return 1

    adapter = OllamaProviderAdapter(client=client)
    runner_config = AgentRunnerConfig(
        agent_name="OM",
        env_prefix="OM",
        debug_env="OM_DEBUG",
        model_env="OM_MODEL",
        initial_debug=DEBUG,
        initial_model=DEFAULT_MODEL,
        prompt_file_env="OM_PROMPT_FILE",
        prompt_file_default=".om_prompt",
        internal_system_prompt=INTERNAL_SYSTEM_PROMPT,
        catch_runtime_errors=True,
    )

    return run_agent_repl(
        adapter=adapter,
        build_tools=build_ollama_tools,
        config=runner_config,
        initial_line=initial_line,
    )


if __name__ == "__main__":
    raise SystemExit(main())

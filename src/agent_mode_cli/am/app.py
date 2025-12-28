from __future__ import annotations

import os
import sys

from agent_mode_cli.core.agent_runner import AgentRunnerConfig, run_agent_repl
from agent_mode_cli.core.cli_help import handle_common_flags
from agent_mode_cli.core.openai_runtime import create_openai_client
from agent_mode_cli.core.openai_adapter import OpenAIProviderAdapter
from agent_mode_cli.core.system_prompt import build_internal_system_prompt
from agent_mode_cli.core.tool_specs import build_openai_tools


DEBUG = os.getenv("AM_DEBUG", "").lower() in ("1", "true", "yes", "on")
DEFAULT_MODEL = os.getenv("AM_MODEL", "gpt-5.1-codex")

INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("AM")
def main(argv: list[str] | None = None) -> int:
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

    adapter = OpenAIProviderAdapter(client=client)
    runner_config = AgentRunnerConfig(
        agent_name="AM",
        env_prefix="AM",
        debug_env="AM_DEBUG",
        model_env="AM_MODEL",
        initial_debug=DEBUG,
        initial_model=DEFAULT_MODEL,
        prompt_file_env="AM_PROMPT_FILE",
        prompt_file_default=".am_prompt",
        internal_system_prompt=INTERNAL_SYSTEM_PROMPT,
        catch_runtime_errors=False,
    )

    return run_agent_repl(
        adapter=adapter,
        build_tools=build_openai_tools,
        config=runner_config,
        initial_line=initial_line,
    )


if __name__ == "__main__":
    raise SystemExit(main())

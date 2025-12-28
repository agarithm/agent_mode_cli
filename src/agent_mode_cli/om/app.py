from __future__ import annotations

import os
import sys

from agent_mode_cli.ai.app import main as ai_main
from agent_mode_cli.core.cli_help import handle_common_flags


def _set_env(key: str, value: str) -> None:
    if not (value or "").strip():
        return
    os.environ[key] = value


def main(argv: list[str] | None = None) -> int:
    """Compatibility wrapper.

    `om` remains as a convenience alias that defaults the unified `ai` REPL
    to the Ollama provider.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_mode_cli import __version__

    flag_exit = handle_common_flags(
        argv,
        usage="om [prompt...]",
        description="Compatibility wrapper around `ai` (defaults provider to Ollama).",
        env_lines=(
            "OM_MODEL        optional (default: gpt-oss)",
            "OM_DEBUG        optional (1/true enables debug)",
            "OM_PROMPT_FILE  optional (default: ~/.om_prompt)",
            "OLLAMA_HOST     optional (use remote Ollama)",
            "AI_PROVIDER     optional (overrides wrapper default)",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit

    _set_env("AI_PROVIDER", "ollama")
    _set_env("AI_MODEL", os.getenv("OM_MODEL", "gpt-oss"))

    if os.getenv("OM_DEBUG"):
        _set_env("AI_DEBUG", os.getenv("OM_DEBUG", ""))
    if os.getenv("OM_PROMPT_FILE"):
        _set_env("AI_PROMPT_FILE", os.getenv("OM_PROMPT_FILE", ""))
    else:
        _set_env("AI_PROMPT_FILE", os.path.join(os.path.expanduser("~"), ".om_prompt"))

    return int(ai_main(argv=argv))


if __name__ == "__main__":
    raise SystemExit(main())

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

    `cm` is a convenience alias that defaults the unified `ai` REPL
    to the GitHub Copilot / GitHub Models provider.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_mode_cli import __version__

    flag_exit = handle_common_flags(
        argv,
        usage="cm [prompt...]",
        description="Compatibility wrapper around `ai` (defaults provider to Copilot / GitHub Models).",
        env_lines=(
            "GITHUB_TOKEN    required (PAT or token with models access)",
            "CM_MODEL        optional (default: google/gemini-latest)",
            "CM_DEBUG        optional (1/true enables debug)",
            "CM_PROMPT_FILE  optional (default: ~/.cm_prompt)",
            "AI_PROVIDER     optional (overrides wrapper default)",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit

    _set_env("AI_PROVIDER", "copilot")
    _set_env("AI_MODEL", os.getenv("CM_MODEL", "google/gemini-latest"))

    if os.getenv("CM_DEBUG"):
        _set_env("AI_DEBUG", os.getenv("CM_DEBUG", ""))
    if os.getenv("CM_PROMPT_FILE"):
        _set_env("AI_PROMPT_FILE", os.getenv("CM_PROMPT_FILE", ""))
    else:
        _set_env("AI_PROMPT_FILE", os.path.join(os.path.expanduser("~"), ".cm_prompt"))

    return int(ai_main(argv=argv))


if __name__ == "__main__":
    raise SystemExit(main())

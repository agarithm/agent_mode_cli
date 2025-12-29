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

    `am` remains as a convenience alias that defaults the unified `ai` REPL
    to the OpenAI provider.
    """

    argv = list(sys.argv[1:] if argv is None else argv)
    from agent_mode_cli import __version__

    flag_exit = handle_common_flags(
        argv,
        usage="am [prompt...]",
        description="Compatibility wrapper around `ai` (defaults provider to OpenAI).",
        env_lines=(
            "OPENAI_API_KEY  required",
            "AM_MODEL        optional (default: gpt-5.1-codex)",
            "AM_DEBUG        optional (1/true enables debug)",
            "AM_PROMPT_FILE  optional (default: ~/.am_prompt)",
            "AI_PROVIDER     optional (overrides wrapper default)",
            "AM_FALLBACKS   optional comma-separated provider list",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit

    # Force the default provider for this alias.
    _set_env("AI_PROVIDER", "openai")
    _set_env("AI_MODEL", os.getenv("AM_MODEL", "gpt-5.1-codex"))

    fallbacks = (os.getenv("AM_FALLBACKS") or "").strip()
    if fallbacks:
        parts = [p.strip().lower() for p in fallbacks.split(",") if p.strip()]
        if len(parts) >= 1:
            _set_env("AI_OPENAI_FALLBACK_PRIMARY", parts[0])
        if len(parts) >= 2:
            _set_env("AI_OPENAI_FALLBACK_SECONDARY", parts[1])

    # Carry debug/prompt config from AM_*, but don't require them.
    if os.getenv("AM_DEBUG"):
        _set_env("AI_DEBUG", os.getenv("AM_DEBUG", ""))
    if os.getenv("AM_PROMPT_FILE"):
        _set_env("AI_PROMPT_FILE", os.getenv("AM_PROMPT_FILE", ""))
    else:
        _set_env("AI_PROMPT_FILE", os.path.join(os.path.expanduser("~"), ".am_prompt"))

    return int(ai_main(argv=argv))


if __name__ == "__main__":
    raise SystemExit(main())

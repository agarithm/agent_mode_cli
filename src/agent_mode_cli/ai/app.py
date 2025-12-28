from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional

from agent_mode_cli.core.ollama_runtime import prepare_runtime


LOG_PREFIX = "[ai]"
DEFAULT_MODEL = os.getenv("AI_MODEL", "gpt-oss")


def _terminal_width() -> int:
    try:
        return shutil.get_terminal_size().columns
    except Exception:
        return 80


def _run_with_optional_glow(cmd: list[str]) -> int:
    glow = shutil.which("glow")
    if not glow:
        proc = subprocess.Popen(cmd)
        return int(proc.wait())

    glow_cmd = [glow, "-w", str(_terminal_width())]
    ollama_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    glow_proc = subprocess.Popen(glow_cmd, stdin=subprocess.PIPE)

    assert ollama_proc.stdout is not None
    assert glow_proc.stdin is not None

    try:
        for chunk in iter(lambda: ollama_proc.stdout.read(4096), b""):
            try:
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            except Exception:
                pass
            try:
                glow_proc.stdin.write(chunk)
                glow_proc.stdin.flush()
            except BrokenPipeError:
                pass
        return int(ollama_proc.wait())
    finally:
        try:
            glow_proc.stdin.close()
        except Exception:
            pass
        try:
            glow_proc.wait(timeout=2)
        except Exception:
            pass


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("-h", "--help"):
        print("usage: ai <prompt...>")
        print("\nZero-shot prompt runner backed by Ollama.")
        return 0
    if argv and argv[0] == "--version":
        from agent_mode_cli import __version__

        print(__version__)
        return 0
    if not argv:
        print("usage: ai <prompt...>")
        return 2

    try:
        prepare_runtime(debug=False, log_prefix=LOG_PREFIX)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    prompt = " ".join(argv)
    prompt += ", I am an expert, try to be brief and concise with examples only when appropriate"

    cmd = ["ollama", "run", DEFAULT_MODEL, prompt]
    return _run_with_optional_glow(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

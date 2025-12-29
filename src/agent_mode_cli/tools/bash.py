from __future__ import annotations

import subprocess


def bash_command(command: str = "", timeout_seconds: int = 30) -> str:
    if not command:
        return "error: command is required"
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
        )
        output = result.stdout or ""
        if result.stderr:
            output += ("\n" if output else "") + result.stderr
        output = output.strip()
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {timeout_seconds} seconds"
    except Exception as exc:  # pragma: no cover - defensive
        return f"error: {exc}"

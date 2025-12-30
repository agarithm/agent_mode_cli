from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Mapping, Optional


_DEFAULT_TIMEOUT_SECONDS = 10
_DEFAULT_MAX_CHARS = 20_000
_DEFAULT_MAX_CODE_CHARS = 200_000


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _truncate(text: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", False
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def python_exec(
    code: str = "",
    *,
    input: Optional[Mapping[str, Any]] = None,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Execute a short Python snippet and return a JSON result.

    - Runs the current interpreter in a subprocess (no shell).
    - Captures stdout/stderr from the snippet.
    - Returns a single JSON object string with:
      ok, returncode, timed_out, duration_ms, stdout, stderr, result, error

    The snippet is executed with two convenience globals:
    - tool_input: the JSON-serializable `input` passed to this tool

    To return a value, assign it to a variable named `result`.
    """

    code = (code or "")
    if not code.strip():
        return _dumps({"ok": False, "error": {"message": "code is required"}})

    if len(code) > _DEFAULT_MAX_CODE_CHARS:
        return _dumps(
            {
                "ok": False,
                "error": {
                    "message": f"code too large (max {_DEFAULT_MAX_CODE_CHARS} chars)",
                    "code_chars": len(code),
                },
            }
        )

    if timeout_seconds <= 0:
        return _dumps({"ok": False, "error": {"message": "timeout_seconds must be > 0"}})

    if max_chars <= 0:
        return _dumps({"ok": False, "error": {"message": "max_chars must be > 0"}})

    input_payload: Any = input if input is not None else None

    wrapper = r"""
import io
import json
import os
import sys
import time
import traceback
import contextlib

start = time.monotonic()

# Read user code from stdin.
user_code = sys.stdin.read()

# Decode tool_input from env.
_raw = os.environ.get("AGENT_MODE_PYTHON_EXEC_INPUT")
try:
    tool_input = json.loads(_raw) if _raw else None
except Exception:
    tool_input = None

stdout_buf = io.StringIO()
stderr_buf = io.StringIO()

globals_dict = {
    "__name__": "__python_exec__",
    "tool_input": tool_input,
}

payload = {
    "ok": False,
    "returncode": 0,
    "timed_out": False,
    "duration_ms": None,
    "stdout": "",
    "stderr": "",
    "result": None,
    "error": None,
}

try:
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        exec(compile(user_code, "<python_exec>", "exec"), globals_dict, globals_dict)
    payload["ok"] = True
    payload["result"] = globals_dict.get("result")
except SystemExit as e:
    # Avoid letting SystemExit kill the wrapper; treat as an error.
    payload["ok"] = False
    payload["returncode"] = int(getattr(e, "code", 1) or 1)
    payload["error"] = {
        "type": "SystemExit",
        "message": str(e),
        "traceback": traceback.format_exc(),
    }
except Exception as e:
    payload["ok"] = False
    payload["returncode"] = 1
    payload["error"] = {
        "type": type(e).__name__,
        "message": str(e),
        "traceback": traceback.format_exc(),
    }

payload["duration_ms"] = int((time.monotonic() - start) * 1000)
payload["stdout"] = stdout_buf.getvalue()
payload["stderr"] = stderr_buf.getvalue()

# Ensure JSON serializability for result.
try:
    json.dumps(payload["result"])
except Exception:
    payload["ok"] = False
    payload["returncode"] = 1
    payload["error"] = {
        "type": "NonJSONResult",
        "message": "result is not JSON-serializable",
        "repr": repr(payload["result"]),
    }
    payload["result"] = None

# Print a single JSON payload.
print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
""".lstrip()

    env = dict(os.environ)
    env["AGENT_MODE_PYTHON_EXEC_INPUT"] = json.dumps(input_payload, ensure_ascii=False, separators=(",", ":"))

    try:
        result = subprocess.run(
            [sys.executable, "-c", wrapper],
            input=code,
            text=True,
            capture_output=True,
            timeout=int(timeout_seconds),
            cwd=os.getcwd(),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return _dumps(
            {
                "ok": False,
                "returncode": None,
                "timed_out": True,
                "duration_ms": int(timeout_seconds * 1000),
                "stdout": "",
                "stderr": "",
                "result": None,
                "error": {"type": "Timeout", "message": f"timed out after {timeout_seconds} seconds"},
            }
        )
    except Exception as exc:  # pragma: no cover - defensive
        return _dumps(
            {
                "ok": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )

    combined_stdout = (result.stdout or "").strip()
    if not combined_stdout:
        # Wrapper always prints JSON; if it didn't, surface stderr.
        err = (result.stderr or "").strip()
        out, trunc = _truncate(err, max_chars=max_chars)
        payload = {
            "ok": False,
            "returncode": result.returncode,
            "timed_out": False,
            "duration_ms": None,
            "stdout": "",
            "stderr": out,
            "result": None,
            "error": {"type": "NoJSON", "message": "python_exec produced no JSON output", "stderr_truncated": trunc},
        }
        return _dumps(payload)

    # Parse JSON from wrapper (it should be the only line). If extra lines exist, keep last line.
    last_line = combined_stdout.splitlines()[-1]
    try:
        payload = json.loads(last_line)
    except Exception:
        stderr = (result.stderr or "").strip()
        out, trunc = _truncate((combined_stdout + "\n" + stderr).strip(), max_chars=max_chars)
        return _dumps(
            {
                "ok": False,
                "returncode": result.returncode,
                "timed_out": False,
                "duration_ms": None,
                "stdout": "",
                "stderr": out,
                "result": None,
                "error": {"type": "InvalidJSON", "message": "python_exec JSON parse failed", "truncated": trunc},
            }
        )

    # Apply truncation to stdout/stderr/traceback fields (post-parse) to keep tool output bounded.
    stdout_text = str(payload.get("stdout") or "")
    stderr_text = str(payload.get("stderr") or "")

    stdout_text, stdout_trunc = _truncate(stdout_text, max_chars=max_chars)
    stderr_text, stderr_trunc = _truncate(stderr_text, max_chars=max_chars)

    payload["stdout"] = stdout_text
    payload["stderr"] = stderr_text
    payload["stdout_truncated"] = bool(stdout_trunc)
    payload["stderr_truncated"] = bool(stderr_trunc)

    err_obj = payload.get("error")
    if isinstance(err_obj, dict) and isinstance(err_obj.get("traceback"), str):
        tb, tb_trunc = _truncate(err_obj["traceback"], max_chars=max_chars)
        err_obj["traceback"] = tb
        err_obj["traceback_truncated"] = bool(tb_trunc)

    return _dumps(payload)

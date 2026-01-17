from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from ._workspace import resolve_workspace_paths, workspace_root


_DEFAULT_MAX_CHARS = 40_000
_DEFAULT_MAX_MATCHES = 200
_DEFAULT_MAX_COLUMNS = 500


def _resolve_paths(paths: Iterable[str]) -> tuple[Optional[List[Path]], Optional[str]]:
    resolved, err = resolve_workspace_paths(paths, root=workspace_root())
    if err:
        return None, err
    assert resolved is not None
    return resolved, None


def search_files(
    query: str = "",
    *,
    paths: Optional[List[str]] = None,
    is_regex: bool = True,
    ignore_case: bool = False,
    glob: Optional[str] = None,
    context_lines: int = 0,
    max_matches: int = _DEFAULT_MAX_MATCHES,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Search for text in files under the current working directory using ripgrep.

    This tool is intended to replace common `rg` usage without requiring the `bash` tool.
    It is read-only and confines all searches to the current working directory.

    Notes:
    - Requires `rg` (ripgrep) to be installed and available on PATH.
    - Exit code 0: matches found
    - Exit code 1: no matches (not an error)
    - Exit code 2: error
    """

    query = (query or "").strip()
    if not query:
        return "error: query is required"

    if max_chars <= 0:
        return "error: max_chars must be > 0"
    if max_matches <= 0:
        return "error: max_matches must be > 0"
    if context_lines < 0:
        return "error: context_lines must be >= 0"

    raw_paths = paths if paths is not None else ["."]
    resolved_paths, err = _resolve_paths(raw_paths)
    if err:
        return err
    assert resolved_paths is not None

    cmd: List[str] = [
        "rg",
        "--no-heading",
        "--with-filename",
        "--line-number",
        "--color",
        "never",
        "--max-count",
        str(int(max_matches)),
        "--max-columns",
        str(int(_DEFAULT_MAX_COLUMNS)),
        "--max-columns-preview",
    ]

    if ignore_case:
        cmd.append("-i")

    if not is_regex:
        cmd.append("-F")

    if context_lines:
        cmd.extend(["-C", str(int(context_lines))])

    if glob is not None and (glob or "").strip():
        # Example: "*.py" or "src/**". This is rg's --glob syntax.
        cmd.extend(["--glob", str(glob)])

    cmd.append("--")
    cmd.append(query)

    for p in resolved_paths:
        cmd.append(str(p))

    try:
        # Keep it robust against binary-ish files and huge repos.
        result = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            cwd=str(workspace_root()),
            timeout=20,
        )
    except FileNotFoundError:
        return "error: rg (ripgrep) is not installed or not on PATH"
    except subprocess.TimeoutExpired:
        return "error: search timed out"
    except Exception as exc:  # pragma: no cover - defensive
        return f"error: {exc}"

    # rg uses exit code 1 for "no matches".
    if result.returncode == 1:
        return "(no matches)"

    if result.returncode not in (0, 1):
        err = (result.stderr or "").strip()
        if err:
            if len(err) > max_chars:
                err = err[:max_chars] + f"\nnote: truncated to max_chars={max_chars}"
            return "error: rg failed\n---\n" + err
        return "error: rg failed"

    out = (result.stdout or "").rstrip()
    if not out:
        return "(no output)"

    if len(out) > max_chars:
        out = out[:max_chars] + f"\nnote: truncated to max_chars={max_chars}"

    return out

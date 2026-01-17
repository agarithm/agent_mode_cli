from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Literal, Optional

import difflib

from ._workspace import resolve_workspace_path, workspace_root


_DEFAULT_MAX_FILE_BYTES = 2_000_000
_DEFAULT_MAX_DIFF_CHARS = 40_000


def _read_text_file(path: Path, *, max_bytes: int = _DEFAULT_MAX_FILE_BYTES) -> tuple[str, Optional[str]]:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return "", f"error: file too large to edit safely (bytes={size}, max={max_bytes})"
    except Exception:
        # If we can't stat, still attempt a bounded read.
        pass

    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        if len(data) > max_bytes:
            return "", f"error: file too large to edit safely (max={max_bytes})"
        return data.decode("utf-8", errors="replace"), None
    except FileNotFoundError:
        return "", None
    except PermissionError:
        return "", "error: permission denied"
    except OSError as exc:
        return "", f"error: {exc}"


def _unified_diff(old: str, new: str, *, from_name: str, to_name: str) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff_iter = difflib.unified_diff(old_lines, new_lines, fromfile=from_name, tofile=to_name)
    diff_text = "".join(diff_iter)
    return diff_text.strip() or "(no changes)"


def _truncate(text: str, max_chars: int = _DEFAULT_MAX_DIFF_CHARS) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", False
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _make_backup(path: Path) -> tuple[Optional[Path], Optional[str]]:
    if not path.exists() or not path.is_file():
        return None, None
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup = path.with_name(path.name + f".bak.{ts}.{os.getpid()}")
    try:
        backup.write_bytes(path.read_bytes())
        return backup, None
    except Exception as exc:
        return None, f"error: failed to create backup - {exc}"


def _validate_ops(edits: Any) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    if edits is None:
        return None, "error: edits is required when mode is 'edits'"
    if not isinstance(edits, list):
        return None, "error: edits must be a list"

    parsed: list[dict[str, Any]] = []
    for idx, raw in enumerate(edits):
        if not isinstance(raw, dict):
            return None, f"error: edits[{idx}] must be an object"
        op = (raw.get("op") or "").strip().lower()
        if not op:
            return None, f"error: edits[{idx}].op is required"
        normalized = dict(raw)
        normalized["op"] = op
        parsed.append(normalized)

    return parsed, None


def _apply_op(text: str, op: dict[str, Any]) -> tuple[str, Optional[str]]:
    count = op.get("count")
    if count is None:
        count = 1
    if not isinstance(count, int) or count < 0:
        return text, "error: count must be an integer >= 0"

    kind = (op.get("op") or "").strip().lower()

    if kind == "replace":
        old = op.get("old")
        new = op.get("new")
        if not isinstance(old, str) or not isinstance(new, str):
            return text, "error: replace requires 'old' and 'new'"
        if old == "":
            return text, "error: replace 'old' must be non-empty"
        occurrences = text.count(old)
        if occurrences == 0:
            return text, "error: replace target not found"
        if count == 0:
            return text.replace(old, new), None
        if occurrences < count:
            return text, f"error: replace expected at least {count} matches, found {occurrences}"
        return text.replace(old, new, count), None

    if kind == "delete":
        old = op.get("old")
        if not isinstance(old, str):
            return text, "error: delete requires 'old'"
        if old == "":
            return text, "error: delete 'old' must be non-empty"
        occurrences = text.count(old)
        if occurrences == 0:
            return text, "error: delete target not found"
        if count == 0:
            return text.replace(old, ""), None
        if occurrences < count:
            return text, f"error: delete expected at least {count} matches, found {occurrences}"
        return text.replace(old, "", count), None

    if kind == "insert_before":
        before = op.get("before")
        content = op.get("content")
        if not isinstance(before, str) or not isinstance(content, str):
            return text, "error: insert_before requires 'before' and 'content'"
        if before == "":
            return text, "error: insert_before 'before' must be non-empty"
        occurrences = text.count(before)
        if occurrences == 0:
            return text, "error: insert_before marker not found"
        if count == 0:
            return text.replace(before, content + before), None
        if occurrences < count:
            return text, f"error: insert_before expected at least {count} matches, found {occurrences}"
        out = text
        for _ in range(count):
            pos = out.find(before)
            if pos < 0:
                break
            out = out[:pos] + content + out[pos:]
        return out, None

    if kind == "insert_after":
        after = op.get("after")
        content = op.get("content")
        if not isinstance(after, str) or not isinstance(content, str):
            return text, "error: insert_after requires 'after' and 'content'"
        if after == "":
            return text, "error: insert_after 'after' must be non-empty"
        occurrences = text.count(after)
        if occurrences == 0:
            return text, "error: insert_after marker not found"
        if count == 0:
            return text.replace(after, after + content), None
        if occurrences < count:
            return text, f"error: insert_after expected at least {count} matches, found {occurrences}"
        out = text
        start = 0
        applied = 0
        while applied < count:
            pos = out.find(after, start)
            if pos < 0:
                break
            insert_at = pos + len(after)
            out = out[:insert_at] + content + out[insert_at:]
            start = insert_at + len(content)
            applied += 1
        return out, None

    return text, f"error: unknown edit op '{kind}'"


def edit_file(
    path: str,
    edits: Any = None,
    *,
    mode: Literal["overwrite", "append", "edits"],
    content: Optional[str] = None,
    dry_run: bool = False,
    make_backup: bool = True,
) -> str:
    """Edit a file within the current working directory.

    Choose a mode to control how the file is modified:
    - mode='overwrite' with content=str
    - mode='append' with content=str
    - mode='edits' with a non-empty `edits` list. Supported ops:
      * replace: {op:'replace', old:str, new:str, count:int? (0 = all)}
      * delete: {op:'delete', old:str, count:int? (0 = all)}
      * insert_before: {op:'insert_before', before:str, content:str, count:int? (0 = all)}
      * insert_after: {op:'insert_after', after:str, content:str, count:int? (0 = all)}
    """

    root = workspace_root()
    resolved, err = resolve_workspace_path(path, root=root)
    if err:
        return err
    assert resolved is not None

    if resolved.exists() and resolved.is_dir():
        return f"error: path is a directory: {path}"

    old_text, err = _read_text_file(resolved)
    if err:
        return err

    normalized_mode = (mode or "").strip().lower()
    if normalized_mode not in {"overwrite", "append", "edits"}:
        return "error: mode must be one of: overwrite, append, edits"

    if normalized_mode in {"overwrite", "append"}:
        if not isinstance(content, str):
            return "error: content is required when mode is 'overwrite' or 'append'"
        new_text = content if normalized_mode == "overwrite" else old_text + content
        edit_summary = f"mode: {normalized_mode}"
    else:
        ops, err = _validate_ops(edits)
        if err:
            return err
        assert ops is not None

        new_text = old_text
        for idx, op in enumerate(ops):
            new_text, op_err = _apply_op(new_text, op)
            if op_err:
                return f"error: edit[{idx}] {op_err}"
        edit_summary = f"mode: edits ({len(ops)} operations)"

    diff = _unified_diff(old_text, new_text, from_name=f"a/{path}", to_name=f"b/{path}")
    diff, was_truncated = _truncate(diff)

    if dry_run:
        suffix = "\nnote: diff truncated" if was_truncated else ""
        return "dry_run: true\n---\n" + diff + suffix

    if make_backup:
        backup_path, backup_err = _make_backup(resolved)
        if backup_err:
            return backup_err
    else:
        backup_path = None

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(new_text, encoding="utf-8")
    except PermissionError:
        return "error: permission denied"
    except OSError as exc:
        return f"error: {exc}"

    header = ["ok: edited file", f"path: {path}", edit_summary]
    if make_backup and backup_path is not None:
        header.append(f"backup: {backup_path.name}")
    if was_truncated:
        header.append("note: diff truncated")
    return "\n".join(header) + "\n---\n" + diff

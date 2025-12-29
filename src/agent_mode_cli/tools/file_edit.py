from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import difflib


_DEFAULT_MAX_FILE_BYTES = 2_000_000
_DEFAULT_MAX_DIFF_CHARS = 40_000


def _workspace_root() -> Path:
    # The agent is intended to operate on the current working directory.
    return Path.cwd().resolve()


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_workspace_path(path: str, *, root: Optional[Path] = None) -> tuple[Optional[Path], Optional[str]]:
    root = _workspace_root() if root is None else root.resolve()
    raw = (path or "").strip()
    if not raw:
        return None, "error: path is required"

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate

    try:
        resolved = candidate.resolve()
    except FileNotFoundError:
        resolved = candidate.absolute()
    except Exception as exc:
        return None, f"error: could not resolve path - {exc}"

    if not _is_within_root(resolved, root):
        return None, "error: path escapes workspace root"
    return resolved, None


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


def write_file(
    path: str,
    content: str,
    *,
    mode: str = "overwrite",
    offset: int = 0,
    dry_run: bool = False,
    make_backup: bool = True,
) -> str:
    """Write content to a file within the current working directory.

    Modes:
    - overwrite: replace the entire file
    - append: append to the end
    - insert: insert at byte offset (default offset=0)

    Returns a unified diff of the change (truncated if needed).
    """

    if offset < 0:
        return "error: offset must be >= 0"

    mode = (mode or "").strip().lower() or "overwrite"
    if mode not in {"overwrite", "append", "insert"}:
        return "error: mode must be one of: overwrite, append, insert"

    root = _workspace_root()
    resolved, err = _resolve_workspace_path(path, root=root)
    if err:
        return err
    assert resolved is not None

    if resolved.exists() and resolved.is_dir():
        return f"error: path is a directory: {path}"

    old_text, err = _read_text_file(resolved)
    if err:
        return err

    # Ensure parent exists (for real writes only).
    if not dry_run:
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return f"error: failed to create parent directories - {exc}"

    new_text: str
    if mode == "overwrite":
        new_text = content
    elif mode == "append":
        new_text = old_text + content
    else:
        # insert
        raw_bytes = old_text.encode("utf-8", errors="replace")
        if offset > len(raw_bytes):
            return f"error: offset out of range (offset={offset}, file_bytes={len(raw_bytes)})"
        insert_bytes = (content or "").encode("utf-8", errors="replace")
        new_bytes = raw_bytes[:offset] + insert_bytes + raw_bytes[offset:]
        new_text = new_bytes.decode("utf-8", errors="replace")

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
        resolved.write_text(new_text, encoding="utf-8")
    except PermissionError:
        return "error: permission denied"
    except OSError as exc:
        return f"error: {exc}"

    header = ["ok: wrote file", f"path: {path}", f"mode: {mode}"]
    if make_backup and backup_path is not None:
        header.append(f"backup: {backup_path.name}")
    if was_truncated:
        header.append("note: diff truncated")
    return "\n".join(header) + "\n---\n" + diff


@dataclass(frozen=True)
class EditOp:
    op: str
    old: Optional[str] = None
    new: Optional[str] = None
    before: Optional[str] = None
    after: Optional[str] = None
    content: Optional[str] = None
    count: Optional[int] = None


def _parse_ops(ops: Any) -> tuple[Optional[list[EditOp]], Optional[str]]:
    if ops is None:
        return None, "error: edits is required"
    if not isinstance(ops, list):
        return None, "error: edits must be a list"

    parsed: list[EditOp] = []
    for idx, raw in enumerate(ops):
        if not isinstance(raw, dict):
            return None, f"error: edits[{idx}] must be an object"
        op = (raw.get("op") or "").strip().lower()
        if not op:
            return None, f"error: edits[{idx}].op is required"
        parsed.append(
            EditOp(
                op=op,
                old=raw.get("old"),
                new=raw.get("new"),
                before=raw.get("before"),
                after=raw.get("after"),
                content=raw.get("content"),
                count=raw.get("count"),
            )
        )
    return parsed, None


def _apply_op(text: str, op: EditOp) -> tuple[str, Optional[str]]:
    count = op.count
    if count is None:
        count = 1
    if not isinstance(count, int) or count < 0:
        return text, "error: count must be an integer >= 0"

    if op.op == "replace":
        if not isinstance(op.old, str) or not isinstance(op.new, str):
            return text, "error: replace requires 'old' and 'new'"
        if op.old == "":
            return text, "error: replace 'old' must be non-empty"
        occurrences = text.count(op.old)
        if occurrences == 0:
            return text, "error: replace target not found"
        if count == 0:
            # replace all
            return text.replace(op.old, op.new), None
        if occurrences < count:
            return text, f"error: replace expected at least {count} matches, found {occurrences}"
        return text.replace(op.old, op.new, count), None

    if op.op == "delete":
        if not isinstance(op.old, str):
            return text, "error: delete requires 'old'"
        if op.old == "":
            return text, "error: delete 'old' must be non-empty"
        occurrences = text.count(op.old)
        if occurrences == 0:
            return text, "error: delete target not found"
        if count == 0:
            return text.replace(op.old, ""), None
        if occurrences < count:
            return text, f"error: delete expected at least {count} matches, found {occurrences}"
        return text.replace(op.old, "", count), None

    if op.op == "insert_before":
        if not isinstance(op.before, str) or not isinstance(op.content, str):
            return text, "error: insert_before requires 'before' and 'content'"
        if op.before == "":
            return text, "error: insert_before 'before' must be non-empty"
        occurrences = text.count(op.before)
        if occurrences == 0:
            return text, "error: insert_before marker not found"
        if count == 0:
            return text.replace(op.before, op.content + op.before), None
        if occurrences < count:
            return text, f"error: insert_before expected at least {count} matches, found {occurrences}"
        out = text
        for _ in range(count):
            pos = out.find(op.before)
            if pos < 0:
                break
            out = out[:pos] + op.content + out[pos:]
        return out, None

    if op.op == "insert_after":
        if not isinstance(op.after, str) or not isinstance(op.content, str):
            return text, "error: insert_after requires 'after' and 'content'"
        if op.after == "":
            return text, "error: insert_after 'after' must be non-empty"
        occurrences = text.count(op.after)
        if occurrences == 0:
            return text, "error: insert_after marker not found"
        if count == 0:
            return text.replace(op.after, op.after + op.content), None
        if occurrences < count:
            return text, f"error: insert_after expected at least {count} matches, found {occurrences}"
        out = text
        start = 0
        applied = 0
        while applied < count:
            pos = out.find(op.after, start)
            if pos < 0:
                break
            insert_at = pos + len(op.after)
            out = out[:insert_at] + op.content + out[insert_at:]
            start = insert_at + len(op.content)
            applied += 1
        return out, None

    return text, f"error: unknown edit op '{op.op}'"


def edit_file(
    path: str,
    edits: Any,
    *,
    dry_run: bool = False,
    make_backup: bool = True,
) -> str:
    """Apply structured edits to a file within the current working directory.

    The `edits` argument is a list of operations. Supported ops:
    - replace: {op:'replace', old:str, new:str, count:int? (0 = all)}
    - delete: {op:'delete', old:str, count:int? (0 = all)}
    - insert_before: {op:'insert_before', before:str, content:str, count:int? (0 = all)}
    - insert_after: {op:'insert_after', after:str, content:str, count:int? (0 = all)}
    """

    root = _workspace_root()
    resolved, err = _resolve_workspace_path(path, root=root)
    if err:
        return err
    assert resolved is not None

    if resolved.exists() and resolved.is_dir():
        return f"error: path is a directory: {path}"

    ops, err = _parse_ops(edits)
    if err:
        return err
    assert ops is not None

    old_text, err = _read_text_file(resolved)
    if err:
        return err

    new_text = old_text
    for idx, op in enumerate(ops):
        new_text, op_err = _apply_op(new_text, op)
        if op_err:
            return f"error: edit[{idx}] {op_err}"

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

    header = ["ok: edited file", f"path: {path}", f"edits: {len(ops)}"]
    if make_backup and backup_path is not None:
        header.append(f"backup: {backup_path.name}")
    if was_truncated:
        header.append("note: diff truncated")
    return "\n".join(header) + "\n---\n" + diff

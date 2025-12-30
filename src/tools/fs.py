from __future__ import annotations

import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_DEFAULT_MAX_LIST_ENTRIES = 2000
_DEFAULT_MAX_READ_BYTES = 200_000


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
        # Resolve can fail on missing paths; still normalize as far as possible.
        resolved = candidate.absolute()
    except Exception as exc:
        return None, f"error: could not resolve path - {exc}"

    # Enforce workspace confinement.
    if not _is_within_root(resolved, root):
        return None, "error: path escapes workspace root"
    return resolved, None


def list_dir(
    path: str = ".",
    *,
    recursive: bool = False,
    max_depth: int = 2,
    max_entries: int = _DEFAULT_MAX_LIST_ENTRIES,
    include_metadata: bool = False,
) -> str:
    """Safely list directories within the current working directory.

    This tool is read-only and refuses to access paths outside the current
    working directory (workspace root).
    """

    if max_entries <= 0:
        return "error: max_entries must be > 0"
    if max_depth < 0:
        return "error: max_depth must be >= 0"

    root = _workspace_root()
    base, err = _resolve_workspace_path(path, root=root)
    if err:
        return err
    assert base is not None

    if not base.exists():
        return f"error: path not found: {path}"
    if not base.is_dir():
        return f"error: not a directory: {path}"

    out: list[str] = []
    truncated = False

    def _safe_stat(p: Path) -> Optional[os.stat_result]:
        try:
            # Avoid following symlinks so we don't accidentally traverse outside root.
            return os.stat(p, follow_symlinks=False)
        except Exception:
            return None

    def _format_entry(p: Path, *, rel: str, is_dir: bool) -> str:
        name = rel + ("/" if is_dir else "")
        if not include_metadata:
            return name

        st = _safe_stat(p)
        if st is None:
            return "?\t?\t?\t" + name

        perms = stat.filemode(st.st_mode)
        size = str(int(st.st_size))
        try:
            mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        except Exception:
            mtime = "?"
        return f"{perms}\t{size}\t{mtime}\t{name}"

    def _append(display: str) -> None:
        nonlocal truncated
        if len(out) >= max_entries:
            truncated = True
            return
        out.append(display)

    try:
        if not recursive:
            entries = sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            for entry in entries:
                if len(out) >= max_entries:
                    truncated = True
                    break
                _append(_format_entry(entry, rel=entry.name, is_dir=entry.is_dir()))
        else:
            # Depth is measured relative to the provided base directory.
            stack: list[tuple[Path, int]] = [(base, 0)]
            while stack:
                current, depth = stack.pop()
                try:
                    entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
                except PermissionError:
                    rel = current.relative_to(base).as_posix() or "."
                    _append(f"{rel}/ (permission denied)")
                    continue

                # For deterministic output, traverse depth-first but keep ordering stable.
                # Push directories in reverse order so they pop in sorted order.
                subdirs: list[Path] = []
                for entry in entries:
                    if len(out) >= max_entries:
                        truncated = True
                        break

                    try:
                        rel = entry.relative_to(base).as_posix()
                    except Exception:
                        # Defensive: should not happen.
                        rel = entry.name

                    is_dir = entry.is_dir()
                    _append(_format_entry(entry, rel=rel, is_dir=is_dir))

                    if is_dir and depth < max_depth:
                        # Avoid following symlinked directories that could escape.
                        if entry.is_symlink():
                            continue
                        resolved_child = None
                        try:
                            resolved_child = entry.resolve()
                        except Exception:
                            continue
                        if not _is_within_root(resolved_child, root):
                            continue
                        subdirs.append(entry)

                if truncated:
                    break
                if depth < max_depth and subdirs:
                    for d in reversed(subdirs):
                        stack.append((d, depth + 1))

    except Exception as exc:
        return f"error: {exc}"

    if not out:
        return "(empty)"
    if truncated:
        out.append(f"note: truncated to max_entries={max_entries}")
    return "\n".join(out)


def read_file(
    path: str,
    *,
    offset: int = 0,
    length: Optional[int] = None,
) -> str:
    """Safely read a file within the current working directory.

    Reads bytes from a file starting at `offset` for `length` bytes.
    If `length` is omitted, reads up to a fixed maximum to keep outputs bounded.
    """

    if offset < 0:
        return "error: offset must be >= 0"
    if length is not None and length <= 0:
        return "error: length must be > 0"

    root = _workspace_root()
    resolved, err = _resolve_workspace_path(path, root=root)
    if err:
        return err
    assert resolved is not None

    if not resolved.exists():
        return f"error: path not found: {path}"
    if resolved.is_dir():
        return f"error: path is a directory: {path}"

    max_bytes = _DEFAULT_MAX_READ_BYTES
    requested = max_bytes if length is None else min(int(length), max_bytes)
    truncated = length is None or (length is not None and int(length) > max_bytes)

    try:
        size = None
        try:
            size = resolved.stat().st_size
        except Exception:
            size = None

        with open(resolved, "rb") as f:
            f.seek(int(offset), os.SEEK_SET)
            data = f.read(int(requested))

        text = data.decode("utf-8", errors="replace")

        header_lines = [
            f"path: {path}",
            f"offset: {int(offset)}",
        ]
        if size is not None:
            header_lines.append(f"file_bytes: {int(size)}")
        header_lines.append(f"returned_bytes: {len(data)}")
        if truncated:
            header_lines.append(f"note: max_returned_bytes={max_bytes}")

        body = text.rstrip("\n")
        return "\n".join(header_lines) + "\n---\n" + (body if body else "(no content)")

    except PermissionError:
        return "error: permission denied"
    except OSError as exc:
        return f"error: {exc}"

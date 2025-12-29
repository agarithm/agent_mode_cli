from __future__ import annotations

import os
import stat
import time
from pathlib import Path
from typing import Optional


def _workspace_root() -> Path:
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


def file_metadata(path: str) -> str:
    """Return basic metadata about a file or directory (read-only)."""

    root = _workspace_root()
    resolved, err = _resolve_workspace_path(path, root=root)
    if err:
        return err
    assert resolved is not None

    try:
        st = os.stat(resolved, follow_symlinks=False)
    except FileNotFoundError:
        return f"error: path not found: {path}"
    except PermissionError:
        return "error: permission denied"
    except OSError as exc:
        return f"error: {exc}"

    mode = st.st_mode
    kind = "dir" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode) else "symlink" if stat.S_ISLNK(mode) else "other"
    perms = stat.filemode(mode)
    mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))
    atime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_atime))
    ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_ctime))

    lines = [
        f"path: {path}",
        f"type: {kind}",
        f"size_bytes: {st.st_size}",
        f"permissions: {perms}",
        f"mtime: {mtime}",
        f"atime: {atime}",
        f"ctime: {ctime}",
    ]
    # Best-effort owner info (may be unavailable in some environments).
    try:
        lines.append(f"uid: {st.st_uid}")
        lines.append(f"gid: {st.st_gid}")
    except Exception:
        pass
    return "\n".join(lines)

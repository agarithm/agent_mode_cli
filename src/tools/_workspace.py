from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def workspace_root() -> Path:
    """Return the workspace root.

    This project treats the current working directory as the workspace root.
    """

    return Path.cwd().resolve()


def is_within_root(candidate: Path, root: Path) -> bool:
    """Return True if candidate is contained within root."""

    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_workspace_path(path: str, *, root: Optional[Path] = None) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve a user-provided path and enforce workspace confinement."""

    root_path = workspace_root() if root is None else root.resolve()
    raw = (path or "").strip()
    if not raw:
        return None, "error: path is required"

    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root_path / candidate

    try:
        resolved = candidate.resolve(strict=False)
    except Exception as exc:
        return None, f"error: could not resolve path - {exc}"

    if not is_within_root(resolved, root_path):
        return None, f"error: path escapes workspace root: {raw}"

    return resolved, None


def resolve_workspace_paths(paths: Iterable[str], *, root: Optional[Path] = None) -> Tuple[Optional[List[Path]], Optional[str]]:
    """Resolve many paths; returns (paths, error)."""

    root_path = workspace_root() if root is None else root.resolve()
    resolved: List[Path] = []

    for raw in paths:
        p = (raw or ".").strip() or "."
        out, err = resolve_workspace_path(p, root=root_path)
        if err:
            return None, err
        assert out is not None
        resolved.append(out)

    return (resolved or [root_path]), None

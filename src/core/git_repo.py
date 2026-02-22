from __future__ import annotations

import os
import subprocess
from typing import Optional


def _run_git(args: list[str], *, cwd: Optional[str] = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def get_repo_root(*, cwd: Optional[str] = None) -> Optional[str]:
    result = _run_git(["rev-parse", "--show-toplevel"], cwd=cwd)
    root = (result.stdout or "").strip()
    return root or None


def is_repo_root(*, cwd: Optional[str] = None) -> bool:
    root = get_repo_root(cwd=cwd)
    if not root:
        return False
    try:
        return os.path.realpath(root) == os.path.realpath(cwd or os.getcwd())
    except Exception:
        return False


def is_git_worktree(*, cwd: Optional[str] = None) -> bool:
    result = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=cwd)
    return result.returncode == 0 and (result.stdout or "").strip().lower() == "true"


def _resolve_git_path(path: str, *, cwd: Optional[str] = None) -> Optional[str]:
    if not path:
        return None
    try:
        if os.path.isabs(path):
            return os.path.realpath(path)
        base = cwd or os.getcwd()
        return os.path.realpath(os.path.join(base, path))
    except Exception:
        return None


def is_linked_worktree(*, cwd: Optional[str] = None) -> bool:
    """Return True for non-primary worktrees created via `git worktree add`."""

    if not is_git_worktree(cwd=cwd):
        return False

    git_dir_res = _run_git(["rev-parse", "--git-dir"], cwd=cwd)
    common_dir_res = _run_git(["rev-parse", "--git-common-dir"], cwd=cwd)
    git_dir = (git_dir_res.stdout or "").strip()
    common_dir = (common_dir_res.stdout or "").strip()
    if not git_dir or not common_dir:
        return False

    resolved_git_dir = _resolve_git_path(git_dir, cwd=cwd)
    resolved_common_dir = _resolve_git_path(common_dir, cwd=cwd)
    if not resolved_git_dir or not resolved_common_dir:
        return False

    return resolved_git_dir != resolved_common_dir


def get_git_branch(*, cwd: Optional[str] = None) -> Optional[str]:
    if not is_git_worktree(cwd=cwd):
        return None

    # Works for normal branches; returns non-zero for detached HEAD.
    result = _run_git(["symbolic-ref", "--short", "-q", "HEAD"], cwd=cwd)
    name = (result.stdout or "").strip()
    return name or None


def get_git_short_sha(*, cwd: Optional[str] = None) -> Optional[str]:
    if not is_git_worktree(cwd=cwd):
        return None
    result = _run_git(["rev-parse", "--short", "HEAD"], cwd=cwd)
    value = (result.stdout or "").strip()
    return value or None


def git_merge(branch: str, *, cwd: Optional[str] = None, no_edit: bool = True) -> tuple[bool, str]:
    args = ["merge"]
    if no_edit:
        args.append("--no-edit")
    args.append(branch)
    result = _run_git(args, cwd=cwd)
    output = (result.stderr or result.stdout or "").strip()
    return (result.returncode == 0, output)

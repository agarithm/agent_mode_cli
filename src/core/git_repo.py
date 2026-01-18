from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
import uuid
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


def get_branch_original(branch: str, *, cwd: Optional[str] = None) -> Optional[str]:
    if not branch:
        return None
    res = _run_git(["config", "--get", f"branch.{branch}.sloppyOriginal"], cwd=cwd)
    value = (res.stdout or "").strip()
    return value or None


def set_branch_original(branch: str, original: str, *, cwd: Optional[str] = None) -> bool:
    if not branch or not original:
        return False
    res = _run_git(["config", f"branch.{branch}.sloppyOriginal", original], cwd=cwd)
    return res.returncode == 0


def get_status_porcelain(*, cwd: Optional[str] = None) -> str:
    if not is_git_worktree(cwd=cwd):
        return ""
    result = _run_git(["status", "--porcelain"], cwd=cwd)
    return (result.stdout or "").strip("\n")


def has_uncommitted_changes(*, cwd: Optional[str] = None) -> bool:
    return bool(get_status_porcelain(cwd=cwd).strip())


def count_commits_ahead(*, base: str, head: str, cwd: Optional[str] = None) -> Optional[int]:
    if not is_git_worktree(cwd=cwd):
        return None
    result = _run_git(["rev-list", "--count", f"{base}..{head}"], cwd=cwd)
    if result.returncode != 0:
        return None
    raw = (result.stdout or "").strip()
    try:
        return int(raw)
    except Exception:
        return None


_SAFE_BRANCH_CHARS = re.compile(r"[^0-9A-Za-z._/-]+")


def _sanitize_branch(name: str) -> str:
    name = name.strip().replace(" ", "-")
    name = _SAFE_BRANCH_CHARS.sub("-", name)
    name = re.sub(r"-+", "-", name)
    name = name.strip("-/.")
    return name


@dataclass(frozen=True)
class AutoBranchResult:
    created: bool
    previous_branch: Optional[str]
    branch: Optional[str]
    reason: Optional[str] = None


def maybe_auto_branch_for_container(
    *,
    cwd: Optional[str] = None,
    env: Optional[dict[str, str]] = None,
) -> AutoBranchResult:
    """Create and checkout an isolated branch when running in-container.

    Controlled by env:
    - AI_CONTAINER_AUTO_BRANCH=0 disables
    - AI_CONTAINER_BRANCH_PREFIX overrides prefix (default: sloppy)
    - AI_CONTAINER_AUTO_BRANCH_DONE=1 skips (idempotency within a process tree)

    Default naming convention is: `<prefix>-yymmdd-<5hex>`.
    """

    env = env or dict(os.environ)
    value = (env.get("AI_CONTAINER_AUTO_BRANCH", "1") or "1").strip().lower()
    if value in ("0", "false", "no", "off"):
        return AutoBranchResult(created=False, previous_branch=None, branch=None, reason="disabled")

    if (env.get("AI_CONTAINER_AUTO_BRANCH_DONE", "") or "").strip().lower() in ("1", "true", "yes", "on"):
        return AutoBranchResult(created=False, previous_branch=None, branch=None, reason="already-done")

    if not is_git_worktree(cwd=cwd):
        return AutoBranchResult(created=False, previous_branch=None, branch=None, reason="not-a-git-repo")

    if not is_repo_root(cwd=cwd):
        return AutoBranchResult(created=False, previous_branch=None, branch=None, reason="not-repo-root")

    previous = get_git_branch(cwd=cwd)
    if previous is None:
        return AutoBranchResult(created=False, previous_branch=None, branch=None, reason="detached-head")

    prefix = (env.get("AI_CONTAINER_BRANCH_PREFIX", "sloppy") or "sloppy").strip().strip("/")
    prefix = _sanitize_branch(prefix) or "sloppy"

    # If user already placed us on an isolation branch (e.g. sloppy-250118-abc12), do nothing.
    # Note: a branch named exactly "sloppy" is NOT treated as an isolation branch.
    if previous.startswith(prefix + "-"):
        original = get_branch_original(previous, cwd=cwd)
        return AutoBranchResult(
            created=False,
            previous_branch=original or previous,
            branch=previous,
            reason="already-on-container-branch",
        )

    day = datetime.now(timezone.utc).strftime("%y%m%d")

    def _branch_exists(name: str) -> bool:
        exists = _run_git(["show-ref", "--verify", "--quiet", f"refs/heads/{name}"], cwd=cwd)
        return exists.returncode == 0

    candidate: Optional[str] = None
    for _ in range(25):
        suffix = uuid.uuid4().hex[:5]
        name = _sanitize_branch(f"{prefix}-{day}-{suffix}")
        if not name:
            continue
        if not _branch_exists(name):
            candidate = name
            break

    if not candidate:
        # Extremely unlikely; fall back to a longer suffix.
        candidate = _sanitize_branch(f"{prefix}-{day}-{uuid.uuid4().hex}")
        if not candidate:
            candidate = f"{prefix}-{day}-{uuid.uuid4().hex}"

    # Create + checkout the new branch.
    # Prefer "git switch -c" but fall back to checkout for older git.
    switch = _run_git(["switch", "-c", candidate], cwd=cwd)
    if switch.returncode != 0:
        checkout = _run_git(["checkout", "-b", candidate], cwd=cwd)
        if checkout.returncode != 0:
            msg = (checkout.stderr or checkout.stdout or "").strip() or "unable to create branch"
            return AutoBranchResult(created=False, previous_branch=previous, branch=previous, reason=msg)

    # Persist the "original" branch for later cleanup.
    set_branch_original(candidate, previous, cwd=cwd)

    return AutoBranchResult(created=True, previous_branch=previous, branch=candidate)


def git_checkout(branch: str, *, cwd: Optional[str] = None, merge: bool = False) -> tuple[bool, str]:
    args = ["checkout"]
    if merge:
        args.append("--merge")
    args.append(branch)
    res = _run_git(args, cwd=cwd)
    out = (res.stderr or res.stdout or "").strip()
    return (res.returncode == 0, out)


def git_add_all(*, cwd: Optional[str] = None) -> tuple[bool, str]:
    res = _run_git(["add", "-A"], cwd=cwd)
    out = (res.stderr or res.stdout or "").strip()
    return (res.returncode == 0, out)


def git_commit(message: str, *, cwd: Optional[str] = None) -> tuple[bool, str]:
    res = _run_git(["commit", "-m", message], cwd=cwd)
    out = (res.stderr or res.stdout or "").strip()
    return (res.returncode == 0, out)


def git_merge(branch: str, *, cwd: Optional[str] = None) -> tuple[bool, str]:
    res = _run_git(["merge", branch], cwd=cwd)
    out = (res.stderr or res.stdout or "").strip()
    return (res.returncode == 0, out)


def git_delete_branch(branch: str, *, cwd: Optional[str] = None, force: bool = False) -> tuple[bool, str]:
    flag = "-D" if force else "-d"
    res = _run_git(["branch", flag, branch], cwd=cwd)
    out = (res.stderr or res.stdout or "").strip()
    return (res.returncode == 0, out)


def git_stash_push(*, cwd: Optional[str] = None, message: str = "ai auto stash") -> tuple[bool, str]:
    # -u includes untracked files.
    res = _run_git(["stash", "push", "-u", "-m", message], cwd=cwd)
    out = (res.stderr or res.stdout or "").strip()
    return (res.returncode == 0, out)


def git_stash_pop(*, cwd: Optional[str] = None) -> tuple[bool, str]:
    res = _run_git(["stash", "pop"], cwd=cwd)
    out = (res.stderr or res.stdout or "").strip()
    return (res.returncode == 0, out)

from __future__ import annotations

import datetime
import os
import re
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import ollama  # noqa: F401

from core.agent_runner import AgentRunnerConfig, ProviderEntry, run_agent_repl
from core.cli_help import handle_common_flags
from core.git_repo import get_repo_root, is_git_worktree, is_linked_worktree
from core.system_prompt import build_internal_system_prompt
from core.ui_runners import run_fullscreen, run_inline
from providers.ollama.adapter import OllamaProviderAdapter
from providers.ollama.runtime import prepare_runtime
from providers.ollama.server import ensure_host_ollama_running, maybe_stop_host_ollama_if_last_container
from providers.ollama.tools import build_tools as build_ollama_tools
from providers.ollama.validation import ensure_ollama_model
from providers.github.adapter import GitHubProviderAdapter
from providers.github.runtime import create_github_models_client
from providers.github.tools import build_tools as build_github_tools
from providers.openai.adapter import OpenAIProviderAdapter
from providers.openai.runtime import create_openai_client
from providers.openai.tools import build_tools as build_openai_tools


INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("AI")
CONTAINER_ENV_FLAG = "AI_IN_CONTAINER"
DEFAULT_CONTAINER_IMAGE = os.getenv("AI_CONTAINER_IMAGE", "localhost/agent-mode-dev:latest")
DEFAULT_PODMAN_BIN = os.getenv("PODMAN_BIN", "docker")

_AGENT_CONTAINER_LABEL = "com.agent_mode.container=1"


def _is_truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def _running_inside_container() -> bool:
    return _is_truthy(os.getenv(CONTAINER_ENV_FLAG, ""))


def _container_launch_disabled() -> bool:
    return _is_truthy(os.getenv("AI_CONTAINER_DISABLE", ""))


_ENV_PASSTHROUGH_DENYLIST = {CONTAINER_ENV_FLAG, "HOME", "OLLAMA_HOST"}

_SAFE_WORKTREE_TOKEN = re.compile(r"[^0-9A-Za-z._-]+")


@dataclass(frozen=True)
class SessionWorktree:
    repo_root: str
    source_cwd: str
    start_branch: str
    worktree_root: str
    session_branch: str
    launch_cwd: str
    carried_state_applied: bool = False
    carried_tracked_paths: tuple[str, ...] = ()
    carried_untracked_paths: tuple[str, ...] = ()


@dataclass(frozen=True)
class DirtyCarryResult:
    applied: bool
    tracked_paths: tuple[str, ...]
    untracked_paths: tuple[str, ...]


def _collect_env_passthrough() -> list[str]:
    flags: list[str] = []
    for key, value in os.environ.items():
        if key in _ENV_PASSTHROUGH_DENYLIST:
            continue
        if "=" in key or "\n" in key or "\x00" in key:
            continue
        if isinstance(value, str) and ("\n" in value or "\x00" in value):
            continue
        flags.extend(["--env", f"{key}={value}"])
    flags.extend(["--env", f"{CONTAINER_ENV_FLAG}=1"])
    return flags


def _normalize_token(value: str) -> str:
    token = _SAFE_WORKTREE_TOKEN.sub("-", (value or "").strip())
    token = re.sub(r"-+", "-", token).strip("-._")
    return token


def _run_git_in_repo(args: list[str], *, repo_root: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_git_in_path(args: list[str], *, path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=path,
        text=True,
        capture_output=True,
        check=False,
    )


def _format_git_error(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "").strip() or "unknown git error"


def _copy_dirty_state_to_worktree(*, repo_root: str, source_cwd: str, worktree_root: str, debug: bool) -> DirtyCarryResult:
    """Copy uncommitted state from source checkout into a fresh session worktree.

    Worktree branches are created from HEAD, so they do not include local dirty
    changes from the original checkout by default. This function replays tracked
    changes via patch and mirrors untracked files so in-session review tools can
    inspect the same local edits the user had before launching `ai`.
    """

    cached_names_res = _run_git_in_repo(["diff", "--cached", "--name-only"], repo_root=repo_root)
    working_names_res = _run_git_in_repo(["diff", "--name-only"], repo_root=repo_root)
    if cached_names_res.returncode != 0 or working_names_res.returncode != 0:
        if debug:
            print("[debug] unable to collect tracked change paths for session carryover", file=sys.stderr)
        return DirtyCarryResult(applied=False, tracked_paths=(), untracked_paths=())

    cached_paths = tuple(p.strip() for p in (cached_names_res.stdout or "").splitlines() if p.strip())
    working_paths = tuple(p.strip() for p in (working_names_res.stdout or "").splitlines() if p.strip())
    tracked_paths = tuple(dict.fromkeys([*cached_paths, *working_paths]))

    cached_diff_res = _run_git_in_repo(["diff", "--cached", "--binary"], repo_root=repo_root)
    working_diff_res = _run_git_in_repo(["diff", "--binary"], repo_root=repo_root)
    if cached_diff_res.returncode != 0 or working_diff_res.returncode != 0:
        if debug:
            print("[debug] unable to collect local diffs for session worktree carryover", file=sys.stderr)
        return DirtyCarryResult(applied=False, tracked_paths=tracked_paths, untracked_paths=())

    cached_patch = cached_diff_res.stdout or ""
    working_patch = working_diff_res.stdout or ""

    patch_applied = True
    if cached_patch.strip():
        apply_cached_res = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=worktree_root,
            input=cached_patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if apply_cached_res.returncode != 0:
            patch_applied = False
            print(
                "warning: could not carry staged tracked changes into the session worktree "
                f"({_format_git_error(apply_cached_res)})",
                file=sys.stderr,
            )
        else:
            if cached_paths:
                stage_res = _run_git_in_path(["add", "--", *cached_paths], path=worktree_root)
            else:
                stage_res = _run_git_in_path(["add", "-A"], path=worktree_root)
            if stage_res.returncode != 0:
                patch_applied = False
                print(
                    "warning: could not stage carried tracked changes in session worktree "
                    f"({_format_git_error(stage_res)})",
                    file=sys.stderr,
                )

    if working_patch.strip():
        apply_working_res = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "-"],
            cwd=worktree_root,
            input=working_patch,
            text=True,
            capture_output=True,
            check=False,
        )
        if apply_working_res.returncode != 0:
            patch_applied = False
            print(
                "warning: could not carry unstaged tracked changes into the session worktree "
                f"({_format_git_error(apply_working_res)})",
                file=sys.stderr,
            )

    untracked_res = _run_git_in_repo(["ls-files", "--others", "--exclude-standard"], repo_root=repo_root)
    if untracked_res.returncode != 0:
        if debug:
            print(f"[debug] unable to list untracked files for carryover: {_format_git_error(untracked_res)}", file=sys.stderr)
        return DirtyCarryResult(applied=patch_applied, tracked_paths=tracked_paths, untracked_paths=())

    untracked_paths = tuple(p.strip() for p in (untracked_res.stdout or "").splitlines() if p.strip())

    copied_count = 0
    for rel in untracked_paths:
        if not rel:
            continue
        src = os.path.join(repo_root, rel)
        dst = os.path.join(worktree_root, rel)
        if not os.path.exists(src):
            continue
        if os.path.isdir(src):
            continue
        try:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied_count += 1
        except Exception as exc:
            if debug:
                print(f"[debug] failed to copy untracked file '{rel}' into session worktree: {exc}", file=sys.stderr)

    if debug and (cached_patch.strip() or working_patch.strip() or copied_count > 0):
        print(
            f"[debug] carried local dirty state into session worktree (cached_patch={'yes' if cached_patch.strip() else 'no'}, working_patch={'yes' if working_patch.strip() else 'no'}, untracked={copied_count})",
            file=sys.stderr,
        )
    return DirtyCarryResult(applied=patch_applied, tracked_paths=tracked_paths, untracked_paths=untracked_paths)


def _safe_repo_join(repo_root: str, rel_path: str) -> Optional[str]:
    candidate = os.path.realpath(os.path.join(repo_root, rel_path))
    repo_real = os.path.realpath(repo_root)
    try:
        if os.path.commonpath([candidate, repo_real]) != repo_real:
            return None
    except Exception:
        return None
    return candidate


def _cleanup_source_checkout_after_merge(session: SessionWorktree, *, debug: bool) -> None:
    if not session.carried_state_applied:
        return

    tracked = [p for p in session.carried_tracked_paths if p]
    if tracked:
        restore_res = _run_git_in_repo(["restore", "--staged", "--worktree", "--", *tracked], repo_root=session.repo_root)
        if restore_res.returncode != 0:
            print(
                "warning: merged, but failed to reconcile tracked files in source checkout "
                f"({_format_git_error(restore_res)})",
                file=sys.stderr,
            )

    for rel in session.carried_untracked_paths:
        if not rel:
            continue
        abs_path = _safe_repo_join(session.repo_root, rel)
        if not abs_path:
            continue
        if not os.path.exists(abs_path):
            continue

        tracked_check = _run_git_in_repo(["ls-files", "--error-unmatch", "--", rel], repo_root=session.repo_root)
        if tracked_check.returncode == 0:
            continue

        try:
            if os.path.isdir(abs_path):
                shutil.rmtree(abs_path)
            else:
                os.remove(abs_path)
        except Exception as exc:
            if debug:
                print(f"[debug] could not remove untracked carryover file '{rel}': {exc}", file=sys.stderr)


def _refresh_source_checkout_after_external_merge(session: SessionWorktree, *, debug: bool) -> None:
    """Refresh source checkout index after branch tip was advanced elsewhere.

    We merge in a temporary worktree, which updates the branch ref without
    updating the source checkout's index. `git status` can then show phantom
    staged entries until the index is refreshed.
    """

    reset_res = _run_git_in_path(["reset", "--mixed", "HEAD"], path=session.source_cwd)
    if reset_res.returncode != 0:
        msg = _format_git_error(reset_res)
        if debug:
            print(f"[debug] source checkout index refresh failed: {msg}", file=sys.stderr)
        else:
            print(f"warning: source checkout index refresh failed ({msg})", file=sys.stderr)


def _is_path_within(path: str, base: str) -> bool:
    try:
        path_real = os.path.realpath(path)
        base_real = os.path.realpath(base)
        return os.path.commonpath([path_real, base_real]) == base_real
    except Exception:
        return False


def _build_extra_mounts(*, launch_cwd: str, volume_suffix: str) -> list[str]:
    """Return additional bind mounts needed for git metadata resolution.

    For linked worktrees, `.git` points to `<main-repo>/.git/worktrees/...` which may
    live outside the worktree path. Mounting `<main-repo>/.git` at the same absolute
    host path makes git commands inside the container resolve correctly.
    """

    if not is_git_worktree(cwd=launch_cwd):
        return []

    common_dir_res = _run_git_in_path(["rev-parse", "--git-common-dir"], path=launch_cwd)
    common_dir_raw = (common_dir_res.stdout or "").strip()
    if common_dir_res.returncode != 0 or not common_dir_raw:
        return []

    if os.path.isabs(common_dir_raw):
        common_dir = os.path.realpath(common_dir_raw)
    else:
        common_dir = os.path.realpath(os.path.join(launch_cwd, common_dir_raw))

    if not os.path.exists(common_dir):
        return []

    if _is_path_within(common_dir, launch_cwd):
        return []

    return ["--volume", f"{common_dir}:{common_dir}{volume_suffix}"]


def _maybe_prepare_session_worktree(*, cwd: str, debug: bool) -> tuple[str, Optional[SessionWorktree]]:
    if not is_git_worktree(cwd=cwd):
        return cwd, None

    if is_linked_worktree(cwd=cwd):
        return cwd, None

    repo_root = get_repo_root(cwd=cwd)
    if not repo_root:
        return cwd, None

    start_branch_result = _run_git_in_repo(["symbolic-ref", "--short", "-q", "HEAD"], repo_root=repo_root)
    start_branch = (start_branch_result.stdout or "").strip()
    if not start_branch:
        if debug:
            print("[debug] detached HEAD; skipping auto-worktree session setup", file=sys.stderr)
        return cwd, None

    try:
        repo_root_real = os.path.realpath(repo_root)
        cwd_real = os.path.realpath(cwd)
        sub_path = os.path.relpath(cwd_real, repo_root_real)
        if sub_path == ".":
            sub_path = ""
    except Exception:
        return cwd, None

    repo_name = _normalize_token(os.path.basename(repo_root_real)) or "repo"
    branch_prefix = "wt"
    parent_dir = os.path.join(os.path.dirname(repo_root_real), ".ai-worktrees")

    base_dir = os.path.join(parent_dir, repo_name)
    os.makedirs(base_dir, exist_ok=True)

    last_error = ""
    for _ in range(25):
        stamp = datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S")
        suffix = uuid.uuid4().hex[:5]
        slug = f"{stamp}-{suffix}"
        branch = f"{branch_prefix}/{slug}"
        worktree_root = os.path.join(base_dir, slug)
        if os.path.exists(worktree_root):
            continue

        add_result = _run_git_in_repo(["worktree", "add", "-b", branch, worktree_root, "HEAD"], repo_root=repo_root_real)
        if add_result.returncode != 0:
            last_error = (add_result.stderr or add_result.stdout or "").strip()
            continue

        launch_cwd = worktree_root if not sub_path else os.path.join(worktree_root, sub_path)
        print(
            f"[ai] session worktree: branch='{branch}' path='{worktree_root}'",
            file=sys.stderr,
        )
        if debug and sub_path:
            print(f"[debug] mapped subdirectory to worktree path: {launch_cwd}", file=sys.stderr)
        carry_result = _copy_dirty_state_to_worktree(
            repo_root=repo_root_real,
            source_cwd=cwd_real,
            worktree_root=worktree_root,
            debug=debug,
        )
        return launch_cwd, SessionWorktree(
            repo_root=repo_root_real,
            source_cwd=cwd_real,
            start_branch=start_branch,
            worktree_root=worktree_root,
            session_branch=branch,
            launch_cwd=launch_cwd,
            carried_state_applied=carry_result.applied,
            carried_tracked_paths=carry_result.tracked_paths,
            carried_untracked_paths=carry_result.untracked_paths,
        )

    if last_error:
        print(f"warning: failed to create session worktree; using current directory ({last_error})", file=sys.stderr)
    return cwd, None


def _merge_session_worktree_back(session: SessionWorktree, *, debug: bool) -> None:
    status = _run_git_in_path(["status", "--porcelain"], path=session.worktree_root)
    if status.returncode != 0:
        print(
            f"warning: cannot inspect session worktree status at '{session.worktree_root}' ({_format_git_error(status)})",
            file=sys.stderr,
        )
        return

    if (status.stdout or "").strip():
        add_res = _run_git_in_path(["add", "-A"], path=session.worktree_root)
        if add_res.returncode != 0:
            print(
                f"warning: auto-merge paused; failed to stage session changes ({_format_git_error(add_res)})",
                file=sys.stderr,
            )
            return
        commit_message = f"ai: auto-commit session {session.session_branch}"
        commit_res = _run_git_in_path(["commit", "-m", commit_message], path=session.worktree_root)
        if commit_res.returncode != 0:
            print(
                f"warning: auto-merge paused; failed to auto-commit session changes ({_format_git_error(commit_res)})",
                file=sys.stderr,
            )
            return

    pre_sync_res = _run_git_in_path(["merge", "--no-edit", session.start_branch], path=session.worktree_root)
    if pre_sync_res.returncode != 0:
        print(
            "warning: pre-merge sync failed; resolve conflicts in the session worktree, then retry manually.",
            file=sys.stderr,
        )
        details = _format_git_error(pre_sync_res)
        if details:
            print(details, file=sys.stderr)
        return
    if debug:
        print(
            f"[debug] pre-merge sync applied: '{session.start_branch}' -> '{session.session_branch}'",
            file=sys.stderr,
        )

    ahead_res = _run_git_in_repo(
        ["rev-list", "--count", f"{session.start_branch}..{session.session_branch}"],
        repo_root=session.repo_root,
    )
    ahead_count = 0
    if ahead_res.returncode == 0:
        try:
            ahead_count = int((ahead_res.stdout or "0").strip() or "0")
        except ValueError:
            ahead_count = 0

    if ahead_count <= 0:
        remove_res = _run_git_in_repo(["worktree", "remove", session.worktree_root], repo_root=session.repo_root)
        if remove_res.returncode != 0 and debug:
            print(f"[debug] worktree cleanup skipped: {_format_git_error(remove_res)}", file=sys.stderr)
        delete_res = _run_git_in_repo(["branch", "-d", session.session_branch], repo_root=session.repo_root)
        if delete_res.returncode != 0 and debug:
            print(f"[debug] branch cleanup skipped: {_format_git_error(delete_res)}", file=sys.stderr)
        return

    merge_slug = _normalize_token(session.session_branch.replace("/", "-")) or "session"
    merge_worktree_root = os.path.join(
        os.path.dirname(session.worktree_root),
        f"__merge-{merge_slug}",
    )

    remove_existing_merge_wt = _run_git_in_repo(
        ["worktree", "remove", merge_worktree_root, "--force"],
        repo_root=session.repo_root,
    )
    if debug and remove_existing_merge_wt.returncode != 0:
        print(f"[debug] pre-clean merge worktree skipped: {_format_git_error(remove_existing_merge_wt)}", file=sys.stderr)

    add_merge_wt = _run_git_in_repo(
        ["worktree", "add", "--force", merge_worktree_root, session.start_branch],
        repo_root=session.repo_root,
    )
    if add_merge_wt.returncode != 0:
        print(
            "warning: auto-merge paused; could not prepare clean merge worktree "
            f"for '{session.start_branch}' ({_format_git_error(add_merge_wt)})",
            file=sys.stderr,
        )
        return

    merge_res = _run_git_in_path(["merge", "--no-edit", session.session_branch], path=merge_worktree_root)
    if merge_res.returncode != 0:
        print(
            "warning: session branch merge requires manual resolution; "
            f"branch='{session.session_branch}' target='{session.start_branch}'",
            file=sys.stderr,
        )
        details = _format_git_error(merge_res)
        if details:
            print(details, file=sys.stderr)
        print(
            f"warning: resolve conflicts in '{merge_worktree_root}', then finalize merge manually.",
            file=sys.stderr,
        )
        return

    remove_merge_wt = _run_git_in_repo(["worktree", "remove", merge_worktree_root], repo_root=session.repo_root)
    if remove_merge_wt.returncode != 0 and debug:
        print(f"[debug] merged, but could not remove temp merge worktree: {_format_git_error(remove_merge_wt)}", file=sys.stderr)

    print(
        f"[ai] merged session branch '{session.session_branch}' into '{session.start_branch}'",
        file=sys.stderr,
    )
    _cleanup_source_checkout_after_merge(session, debug=debug)
    _refresh_source_checkout_after_external_merge(session, debug=debug)
    remove_res = _run_git_in_repo(["worktree", "remove", session.worktree_root], repo_root=session.repo_root)
    if remove_res.returncode != 0:
        print(
            f"warning: merged, but failed to remove worktree '{session.worktree_root}' ({_format_git_error(remove_res)})",
            file=sys.stderr,
        )
    delete_res = _run_git_in_repo(["branch", "-d", session.session_branch], repo_root=session.repo_root)
    if delete_res.returncode != 0:
        print(
            f"warning: merged, but failed to delete session branch '{session.session_branch}' ({_format_git_error(delete_res)})",
            file=sys.stderr,
        )


def _maybe_run_inside_container(argv: list[str]) -> None:
    if _running_inside_container() or _container_launch_disabled():
        return

    podman_bin = os.getenv("PODMAN_BIN", DEFAULT_PODMAN_BIN)
    if shutil.which(podman_bin) is None:
        print(
            f"warning: '{podman_bin}' not found; running natively without container",
            file=sys.stderr,
        )
        return

    cwd = os.getcwd()
    if not cwd:
        return

    container_image = os.getenv("AI_CONTAINER_IMAGE", DEFAULT_CONTAINER_IMAGE)
    stdio_flags = ["--interactive"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        stdio_flags.append("--tty")

    # Docker-specific vs Podman-specific flags
    is_podman = "podman" in podman_bin.lower()
    userns_flags = ["--userns=keep-id"] if is_podman else ["--user", f"{os.getuid()}:{os.getgid()}"]
    volume_suffix = ":rw,Z" if is_podman else ":rw"

    debug = os.getenv("AI_DEBUG", "").lower() in ("1", "true", "yes", "on")
    launch_cwd, session_worktree = _maybe_prepare_session_worktree(cwd=cwd, debug=debug)
    ensure_host_ollama_running(debug=debug)

    session_env_flags: list[str] = []
    if session_worktree is not None:
        session_env_flags.extend(
            [
                "--env",
                f"AI_SESSION_BASE_BRANCH={session_worktree.start_branch}",
                "--env",
                f"AI_SESSION_BRANCH={session_worktree.session_branch}",
                "--env",
                f"AI_SESSION_WORKTREE_ROOT={session_worktree.worktree_root}",
            ]
        )

    extra_mount_flags = _build_extra_mounts(launch_cwd=launch_cwd, volume_suffix=volume_suffix)

    cmd = [
        podman_bin,
        "run",
        "--rm",
        "--label",
        _AGENT_CONTAINER_LABEL,
        "--network",
        "host",
        *userns_flags,
        *stdio_flags,
        *_collect_env_passthrough(),
        *session_env_flags,
        *extra_mount_flags,
        "--volume",
        f"{launch_cwd}:{launch_cwd}{volume_suffix}",
        "--workdir",
        launch_cwd,
        container_image,
        "ai",
        *argv,
    ]

    print(
        f"[ai] launching container '{container_image}' via {podman_bin} for {launch_cwd}",
        file=sys.stderr,
    )
    result: subprocess.CompletedProcess[str] | None = None
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        print(
            f"error: unable to execute '{podman_bin}' for container launch",
            file=sys.stderr,
        )
        raise
    except KeyboardInterrupt:
        # Still attempt teardown in finally; container may still be running.
        result = subprocess.CompletedProcess(cmd, 130)  # type: ignore[arg-type]
    finally:
        if session_worktree is not None:
            try:
                _merge_session_worktree_back(session_worktree, debug=debug)
            except Exception as exc:
                print(f"warning: session merge failed unexpectedly ({exc})", file=sys.stderr)
        try:
            maybe_stop_host_ollama_if_last_container(podman_bin, _AGENT_CONTAINER_LABEL, debug=debug)
        except Exception as exc:
            if debug:
                print(f"[debug] host ollama shutdown check failed: {exc}", file=sys.stderr)

    raise SystemExit((result.returncode if result is not None else 1))


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _maybe_run_inside_container(argv)
    from version import __version__

    def _extract_inline_flag(args: list[str]) -> tuple[bool, list[str]]:
        inline = False
        remaining: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            if token == "--inline":
                inline = True
                i += 1
                continue
            remaining.append(token)
            i += 1
        return inline, remaining

    inline, argv = _extract_inline_flag(argv)

    debug = os.getenv("AI_DEBUG", "").lower() in ("1", "true", "yes", "on")
    env_provider = os.getenv("AI_PROVIDER", "ollama").strip().lower() or "ollama"
    base_ollama_model = os.getenv("AI_MODEL", "gpt-oss:latest")

    flag_exit = handle_common_flags(
        argv,
        usage="ai [--inline] [provider] [model] [prompt...]",
        description=(
            "Agent-mode CLI (full REPL) with dynamic provider switching. "
            "Default provider is Ollama."
        ),
        env_lines=(
            "AI_PROVIDER     optional (default: ollama)",
            "AI_MODEL        optional (default: gpt-oss:latest; applies to current provider)",
            "AI_DEBUG        optional (1/true enables debug)",
            "AI_PROMPT_FILE  optional (default: ~/.ai_prompt)",
            "OPENAI_API_KEY  required for OpenAI provider",
            "GITHUB_TOKEN    required for GitHub Models provider",
            "First CLI args  optional magic keywords for provider/model",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit

    def _create_ollama_adapter() -> OllamaProviderAdapter:
        try:
            import ollama as _ollama  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ollama Python package is not available. Run inside the dev container or install it manually."
            ) from exc
        client = _ollama.Client()
        return OllamaProviderAdapter(client=client)

    def _create_openai_adapter() -> OpenAIProviderAdapter:
        client = create_openai_client()
        return OpenAIProviderAdapter(client=client)

    def _create_github_adapter() -> GitHubProviderAdapter:
        client = create_github_models_client()
        return GitHubProviderAdapter(client=client)

    def _build_providers(default_ollama_model: str, overrides: dict[str, str] | None = None) -> dict[str, ProviderEntry]:
        overrides = overrides or {}
        return {
            "ollama": ProviderEntry(
                name="ollama",
                description="Ollama (default)",
                default_model=overrides.get("ollama") or default_ollama_model,
                build_tools=build_ollama_tools,
                create_adapter=_create_ollama_adapter,
                prepare_runtime=lambda debug: prepare_runtime(debug=debug, log_prefix="[ai]"),
                validate_model=lambda model, debug: ensure_ollama_model(model, debug=debug, log_prefix="[ai]"),
            ),
            "github": ProviderEntry(
                name="github",
                description="GitHub Models (requires GITHUB_TOKEN)",
                default_model=overrides.get("github") or os.getenv("AI_GITHUB_MODEL", "xai/grok-3"),
                build_tools=build_github_tools,
                create_adapter=_create_github_adapter,
                prepare_runtime=None,
                fallback_providers=(
                    os.getenv("AI_GITHUB_FALLBACK_PRIMARY", "openai"),
                    os.getenv("AI_GITHUB_FALLBACK_SECONDARY", "ollama"),
                ),
            ),
            "openai": ProviderEntry(
                name="openai",
                description="OpenAI (requires OPENAI_API_KEY)",
                default_model=overrides.get("openai") or os.getenv("AI_OPENAI_MODEL", "gpt-5.2"),
                build_tools=build_openai_tools,
                create_adapter=_create_openai_adapter,
                prepare_runtime=None,
                fallback_providers=(
                    os.getenv("AI_OPENAI_FALLBACK_PRIMARY", "github"),
                    os.getenv("AI_OPENAI_FALLBACK_SECONDARY", "ollama"),
                ),
            ),
        }

    cli_prevalidated_models: dict[str, str] = {}
    cli_adapter_cache: dict[str, object] = {}

    def _extract_cli_overrides(raw_args: list[str], providers_map: dict[str, ProviderEntry]) -> tuple[Optional[str], Optional[str], list[str]]:
        remaining = list(raw_args)
        if not remaining:
            return None, None, []

        candidate_provider = (remaining[0] or "").strip().lower()
        if candidate_provider not in providers_map:
            return None, None, remaining

        provider_override = candidate_provider
        tokens_after_provider = remaining[1:]

        if not tokens_after_provider:
            return provider_override, None, []

        candidate_model = tokens_after_provider[0]

        def _candidate_is_valid_model(provider_name: str, candidate: str) -> tuple[bool, Optional[str]]:
            entry = providers_map[provider_name]
            normalized = (candidate or "").strip()
            if not normalized:
                return False, None
            if entry.validate_model is not None:
                try:
                    validated = entry.validate_model(normalized, debug)
                    cli_prevalidated_models[provider_name] = validated
                    return True, validated
                except Exception as exc:
                    if debug:
                        print(
                            f"[debug] CLI model '{normalized}' rejected for provider '{provider_name}': {exc}",
                            file=sys.stderr,
                        )
                    return False, None

            adapter = cli_adapter_cache.get(provider_name)
            if adapter is None:
                try:
                    if entry.prepare_runtime is not None:
                        entry.prepare_runtime(debug)
                    adapter = entry.create_adapter()
                    cli_adapter_cache[provider_name] = adapter
                except Exception as exc:
                    if debug:
                        print(
                            f"[debug] CLI could not initialize adapter for provider '{provider_name}': {exc}",
                            file=sys.stderr,
                        )
                    return False, None
            try:
                available = [m.strip() for m in adapter.list_models(debug=debug)]  # type: ignore[attr-defined]
            except Exception as exc:
                if debug:
                    print(
                        f"[debug] CLI could not list models for provider '{provider_name}': {exc}",
                        file=sys.stderr,
                    )
                return False, None
            available = [m for m in available if m]
            if normalized in available:
                return True, normalized
            if debug:
                print(
                    f"[debug] CLI candidate '{normalized}' not found in provider '{provider_name}' catalog",
                    file=sys.stderr,
                )
            return False, None

        is_valid_model, normalized_model = _candidate_is_valid_model(provider_override, candidate_model)
        if is_valid_model and normalized_model:
            return provider_override, normalized_model, tokens_after_provider[1:]
        return provider_override, None, tokens_after_provider

    providers = _build_providers(base_ollama_model)
    provider_override, model_override, prompt_tokens = _extract_cli_overrides(argv, providers)

    provider_model_overrides: dict[str, str] = {}
    if provider_override and model_override:
        provider_model_overrides[provider_override] = model_override

    if provider_override:
        os.environ["AI_PROVIDER"] = provider_override
    if provider_override and model_override:
        os.environ["AI_MODEL"] = model_override

    initial_provider = provider_override or env_provider
    default_ollama_model = cli_prevalidated_models.get("ollama") or base_ollama_model

    if initial_provider == "ollama" and "ollama" not in cli_prevalidated_models:
        try:
            ensured_model = ensure_ollama_model(
                provider_model_overrides.get("ollama") or default_ollama_model,
                debug=debug,
                log_prefix="[ai]",
            )
            default_ollama_model = ensured_model
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    providers = _build_providers(default_ollama_model, overrides=provider_model_overrides)
    initial_line = " ".join(prompt_tokens) if prompt_tokens else None

    runner_config = AgentRunnerConfig(
        agent_name="AI",
        env_prefix="AI",
        debug_env="AI_DEBUG",
        model_env="AI_MODEL",
        initial_debug=debug,
        initial_model=default_ollama_model,
        prompt_file_env="AI_PROMPT_FILE",
        prompt_file_default=".ai_prompt",
        internal_system_prompt=INTERNAL_SYSTEM_PROMPT,
        catch_runtime_errors=True,
    )

    initial_model_banner = (os.getenv("AI_MODEL") or "").strip()
    if not initial_model_banner:
        entry = providers.get(initial_provider)
        initial_model_banner = (entry.default_model if entry else "") or "unknown"
    print(
        f"[ai] Provider: {initial_provider} {initial_model_banner}",
    )

    try:
        repl_runner = run_inline if inline else run_fullscreen
        return run_agent_repl(
            providers=providers,
            initial_provider=initial_provider,
            config=runner_config,
            initial_line=initial_line,
            repl_runner=repl_runner,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

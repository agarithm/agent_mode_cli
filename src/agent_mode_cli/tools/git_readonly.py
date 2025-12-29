from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import difflib


_DEFAULT_MAX_CHARS = 40_000
_DEFAULT_MAX_UNTRACKED_FILES = 20
_DEFAULT_MAX_UNTRACKED_FILE_BYTES = 200_000


def _workspace_root() -> Path:
    # The agent is intended to operate on the current working directory.
    return Path.cwd().resolve()


def _is_within_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _truncate(text: str, *, max_chars: int = _DEFAULT_MAX_CHARS) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", False
    if len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def _coerce_paths(paths: Any) -> tuple[Optional[list[str]], Optional[str]]:
    if paths is None:
        return None, None
    if isinstance(paths, str):
        p = paths.strip()
        return ([p] if p else []), None
    if isinstance(paths, list):
        out: list[str] = []
        for idx, item in enumerate(paths):
            if not isinstance(item, str):
                return None, f"error: paths[{idx}] must be a string"
            s = item.strip()
            if s:
                out.append(s)
        return out, None
    return None, "error: paths must be a string, a list of strings, or null"


def _validate_paths_within_root(paths: Sequence[str], *, root: Path) -> tuple[list[str], Optional[str]]:
    validated: list[str] = []
    for raw in paths:
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate.absolute()
        except Exception as exc:
            return [], f"error: could not resolve path '{raw}' - {exc}"
        if not _is_within_root(resolved, root):
            return [], "error: path escapes workspace root"
        # git expects paths relative to repo root; our repo root is root (enforced below).
        try:
            rel = resolved.relative_to(root).as_posix() or "."
        except Exception:
            rel = raw
        validated.append(rel)
    return validated, None


def _run_git(args: Iterable[str], *, cwd: Path) -> tuple[Optional[str], Optional[str]]:
    try:
        result = subprocess.run(
            ["git", "--no-pager", *list(args)],
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=20,
        )
        out = (result.stdout or "")
        err = (result.stderr or "")
        combined = out + ("\n" + err if err.strip() else "")
        combined = combined.strip() or "(no output)"
        if result.returncode != 0:
            return None, combined
        return combined, None
    except FileNotFoundError:
        return None, "error: git executable not found"
    except subprocess.TimeoutExpired:
        return None, "error: git command timed out"
    except Exception as exc:
        return None, f"error: {exc}"


def _git_toplevel(*, root: Path) -> tuple[Optional[Path], Optional[str]]:
    out, err = _run_git(["rev-parse", "--show-toplevel"], cwd=root)
    if err:
        return None, "error: not a git repository"
    toplevel = (out or "").splitlines()[0].strip()
    if not toplevel:
        return None, "error: not a git repository"
    try:
        tl = Path(toplevel).resolve()
    except Exception:
        return None, "error: could not resolve git toplevel"
    # Prevent git from walking outside the workspace root.
    if not _is_within_root(tl, root) and tl != root:
        return None, "error: git repository root is outside workspace root"
    if tl != root:
        # We only allow git operations when the repo root is exactly the workspace root
        # to avoid reading parent dirs outside the agent's sandbox.
        return None, "error: git repository root must be the workspace root"
    return tl, None


def _git_untracked_files(*, root: Path) -> tuple[Optional[list[str]], Optional[str]]:
    out, err = _run_git(["ls-files", "--others", "--exclude-standard"], cwd=root)
    if err:
        return None, err
    files = [line.strip() for line in (out or "").splitlines() if line.strip()]
    return files, None


def _filter_untracked_by_prefix(untracked: Sequence[str], paths: Optional[Sequence[str]]) -> list[str]:
    if not paths:
        return list(untracked)
    prefixes = []
    for p in paths:
        pp = (p or "").strip().rstrip("/")
        if pp and pp != ".":
            prefixes.append(pp + "/")
            prefixes.append(pp)
    if not prefixes:
        return list(untracked)
    out: list[str] = []
    for f in untracked:
        for pref in prefixes:
            if f == pref or f.startswith(pref if pref.endswith("/") else pref + "/"):
                out.append(f)
                break
    return out


def _untracked_as_diff(path: Path, rel: str, *, max_bytes: int) -> str:
    try:
        with open(path, "rb") as f:
            data = f.read(max_bytes + 1)
        truncated_bytes = len(data) > max_bytes
        if truncated_bytes:
            data = data[:max_bytes]
        text = data.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return f"# note: untracked file disappeared: {rel}\n"
    except PermissionError:
        return f"# note: permission denied reading untracked file: {rel}\n"
    except OSError as exc:
        return f"# note: error reading untracked file {rel}: {exc}\n"

    new_lines = text.splitlines(keepends=True)
    diff = "".join(
        difflib.unified_diff(
            [],
            new_lines,
            fromfile="/dev/null",
            tofile=f"b/{rel}",
        )
    )
    if truncated_bytes:
        diff = (diff.rstrip() + f"\n# note: content truncated to {max_bytes} bytes\n")
    return diff.strip() + "\n"


def git_status(*, max_chars: int = _DEFAULT_MAX_CHARS) -> str:
    """Read-only git status (porcelain) confined to the current working directory."""

    root = _workspace_root()
    _, err = _git_toplevel(root=root)
    if err:
        return err

    out, err = _run_git(["status", "--porcelain=v1", "-b"], cwd=root)
    if err:
        return err
    text, truncated = _truncate(out or "", max_chars=max_chars)
    if truncated:
        return text + f"\nnote: truncated to max_chars={max_chars}"
    return text


def git_diff(
    paths: Any = None,
    *,
    staged: bool = False,
    include_untracked: bool = True,
    max_untracked_files: int = _DEFAULT_MAX_UNTRACKED_FILES,
    max_untracked_file_bytes: int = _DEFAULT_MAX_UNTRACKED_FILE_BYTES,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> str:
    """Read-only git diff for the workspace repo.

    If `paths` is provided, it must be a string or list of strings and will be
    validated to stay within the workspace root.
    """

    root = _workspace_root()
    _, err = _git_toplevel(root=root)
    if err:
        return err

    coerced, err = _coerce_paths(paths)
    if err:
        return err
    path_args: list[str] = []
    validated_paths: Optional[list[str]] = None
    if coerced is not None and coerced:
        validated, err = _validate_paths_within_root(coerced, root=root)
        if err:
            return err
        validated_paths = validated
        path_args = ["--", *validated]

    diff_args: list[str] = ["diff"]
    if staged:
        diff_args.append("--staged")
    diff_args.extend(path_args)

    out, err = _run_git(diff_args, cwd=root)
    if err:
        return err

    combined_parts: list[str] = []
    base_diff = (out or "").strip()
    if base_diff and base_diff != "(no output)":
        combined_parts.append(base_diff)

    untracked_note: Optional[str] = None
    if include_untracked:
        if max_untracked_files <= 0:
            return "error: max_untracked_files must be > 0"
        if max_untracked_file_bytes <= 0:
            return "error: max_untracked_file_bytes must be > 0"

        untracked, err = _git_untracked_files(root=root)
        if err:
            return err
        assert untracked is not None
        filtered = _filter_untracked_by_prefix(untracked, validated_paths)
        if filtered:
            filtered = sorted(filtered)
            if len(filtered) > max_untracked_files:
                untracked_note = f"note: untracked files truncated to max_untracked_files={max_untracked_files}"
                filtered = filtered[:max_untracked_files]

            for rel in filtered:
                p = (root / rel)
                # Skip directories.
                try:
                    if p.is_dir():
                        continue
                except Exception:
                    continue
                combined_parts.append(_untracked_as_diff(p, rel, max_bytes=max_untracked_file_bytes).strip())

    combined = "\n\n".join([p for p in combined_parts if p.strip()])
    if not combined.strip():
        combined = "(no changes)"
    if untracked_note:
        combined = combined + "\n" + untracked_note

    text, truncated = _truncate(combined, max_chars=max_chars)
    if truncated:
        return text + f"\nnote: truncated to max_chars={max_chars}"
    return text

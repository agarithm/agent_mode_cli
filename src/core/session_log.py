from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if is_dataclass(value):
        return _safe_json(asdict(value))
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(v) for v in value]
    return str(value)


def _find_git_root(start: Path, *, max_depth: int = 20) -> Optional[Path]:
    current = start
    try:
        current = current.absolute()
    except Exception:
        pass

    for _ in range(max_depth):
        marker = current / ".git"
        try:
            # Prefer is_dir() so we don't match weird files named ".git".
            if marker.is_dir() or marker.exists():
                return current
        except Exception:
            # Permission/mount boundary issues: treat as not-a-repo.
            return None

        if current.parent == current:
            break
        current = current.parent
    return None


def _ensure_gitignore_has_pattern(git_root: Path, pattern: str) -> None:
    path = git_root / ".gitignore"
    try:
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
    except Exception:
        existing = ""

    normalized_lines = [ln.strip() for ln in existing.splitlines()]
    if pattern.strip() in normalized_lines:
        return

    try:
        prefix = "" if existing.endswith("\n") or existing == "" else "\n"
        suffix = "\n" if not existing.endswith("\n") else ""
        with path.open("a", encoding="utf-8") as f:
            if prefix:
                f.write(prefix)
            f.write(pattern.rstrip() + "\n")
            if suffix:
                f.write(suffix)
    except Exception:
        return


class SessionLogger:
    """Append-only session event logger.

    Canonical format is JSON Lines (JSONL) so it is:
    - human readable (plain text)
    - DuckDB readable via read_json_auto('.../*.jsonl')

    It also maintains a lightweight Markdown transcript for quick browsing.

    Logging must never crash the REPL; all write errors are swallowed.
    """

    def __init__(
        self,
        *,
        root_dir: Path,
        session_id: str,
        enabled: bool = True,
        max_inline_chars: int = 2_000_000,
        debug: bool = False,
    ) -> None:
        self.enabled = enabled
        self.debug = debug
        self.session_id = session_id
        self.root_dir = root_dir
        self.session_dir = root_dir / "sessions" / session_id
        self.events_path = self.session_dir / "events.jsonl"
        self.transcript_path = self.session_dir / "transcript.md"
        self.meta_path = self.session_dir / "meta.json"
        self.blobs_dir = self.session_dir / "blobs"
        self._lock = threading.Lock()
        self._seq = 0
        self.max_inline_chars = max(1, int(max_inline_chars))

        if not self.enabled:
            return

        try:
            self.blobs_dir.mkdir(parents=True, exist_ok=True)
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self.root_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If this fails we just disable logging silently.
            self.enabled = False
            return

        # If we're operating inside a git repo, make sure .sloppy is ignored.
        # This is intentionally best-effort and can be disabled.
        if (os.getenv("AI_SLOPPY_GITIGNORE") or "").strip().lower() not in {"0", "false", "off", "no"}:
            try:
                # Start discovery from the logging root itself so we don't accidentally
                # traverse outside the mounted project directory in container scenarios.
                git_root = _find_git_root(self.root_dir)
                if git_root is not None:
                    try:
                        root_resolved = self.root_dir.absolute()
                        git_resolved = git_root.absolute()
                        try:
                            root_resolved.relative_to(git_resolved)
                            in_repo = True
                        except Exception:
                            in_repo = False
                    except Exception:
                        in_repo = False

                    # Only touch .gitignore if the logging directory is actually in the repo.
                    if in_repo:
                        _ensure_gitignore_has_pattern(git_root, ".sloppy/")
            except Exception:
                pass

    @staticmethod
    def default_root_dir() -> Path:
        override = (os.getenv("AI_SLOPPY_DIR") or "").strip()
        if override:
            return Path(override).expanduser()
        return Path(os.getcwd()) / ".sloppy"

    @staticmethod
    def is_enabled_from_env() -> bool:
        value = (os.getenv("AI_SLOPPY_LOG") or "").strip().lower()
        if value in {"0", "false", "off", "no"}:
            return False
        return True

    @staticmethod
    def new_session_id(prefix: str = "session") -> str:
        # Timestamped + short random suffix to avoid collisions.
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        rnd = os.urandom(4).hex()
        return f"{prefix}_{ts}_{rnd}"

    def _spill_text(self, text: str, *, kind: str) -> Dict[str, Any]:
        data = text.encode("utf-8", errors="replace")
        digest = sha256(data).hexdigest()
        rel = Path("blobs") / f"{kind}_{digest}.txt"
        path = self.session_dir / rel
        try:
            if not path.exists():
                path.write_bytes(data)
        except Exception:
            # Fall back to inline data if we cannot spill.
            return {"inline": text}
        return {
            "blob": str(rel.as_posix()),
            "sha256": digest,
            "chars": len(text),
        }

    def _maybe_inline_text(self, text: str, *, kind: str) -> Dict[str, Any]:
        if len(text) <= self.max_inline_chars:
            return {"inline": text}
        return self._spill_text(text, kind=kind)

    def write_meta(self, meta: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            payload = {"session_id": self.session_id, "written_at": _utc_now_iso(), **_safe_json(meta)}
            self.meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            return

    def log_event(self, event: str, data: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._seq += 1
            seq = self._seq

            record: Dict[str, Any] = {
                "ts": _utc_now_iso(),
                "session_id": self.session_id,
                "seq": seq,
                "event": event,
                "data": _safe_json(data or {}),
            }

            try:
                self.session_dir.mkdir(parents=True, exist_ok=True)
                with self.events_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                return

    def log_transcript(self, role: str, content: str, *, tool_name: Optional[str] = None) -> None:
        if not self.enabled:
            return

        header = role
        if tool_name:
            header = f"{role} ({tool_name})"
        block = f"## {_utc_now_iso()} {header}\n\n{content.rstrip()}\n\n"

        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            with self.transcript_path.open("a", encoding="utf-8") as f:
                f.write(block)
        except Exception:
            return

    def log_chat_message(
        self,
        *,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        if not self.enabled:
            return

        content_info = self._maybe_inline_text(content or "", kind=f"msg_{role}")
        data: Dict[str, Any] = {
            "role": role,
            "content": content_info,
        }
        if tool_name:
            data["tool_name"] = tool_name
        if tool_call_id:
            data["tool_call_id"] = tool_call_id
        if tool_calls is not None:
            data["tool_calls"] = tool_calls

        self.log_event("chat_message", data)

        # Human transcript uses inline content only. If it spilled, we add a pointer.
        if "inline" in content_info:
            transcript_text = content_info["inline"]
        else:
            transcript_text = f"(see {content_info.get('blob')}; sha256={content_info.get('sha256')})"
        self.log_transcript(role, transcript_text, tool_name=tool_name)

    def log_tool_call(self, *, name: str, arguments: Dict[str, Any], call_id: Optional[str]) -> None:
        self.log_event(
            "tool_call",
            {
                "name": name,
                "call_id": call_id,
                "arguments": _safe_json(arguments),
            },
        )

    def log_tool_result(self, *, name: str, call_id: Optional[str], output_text: str) -> None:
        output_info = self._maybe_inline_text(output_text or "", kind=f"tool_{name}")
        self.log_event(
            "tool_result",
            {
                "name": name,
                "call_id": call_id,
                "output": output_info,
            },
        )
        if "inline" in output_info:
            transcript_text = output_info["inline"]
        else:
            transcript_text = f"(see {output_info.get('blob')}; sha256={output_info.get('sha256')})"
        self.log_transcript("tool", transcript_text, tool_name=name)

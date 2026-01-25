from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:  # Windows compatibility
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None  # type: ignore


_HOST_OLLAMA_URL = "http://127.0.0.1:11434"
_HOST_OLLAMA_VERSION_URL = _HOST_OLLAMA_URL + "/api/version"


def host_ollama_reachable(timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(_HOST_OLLAMA_VERSION_URL, timeout=timeout) as resp:
            return 200 <= getattr(resp, "status", 200) < 300
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def _agent_state_dir() -> Path:
    base = os.getenv("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return Path(base) / "agent-mode-cli"


def _ollama_pid_path() -> Path:
    return _agent_state_dir() / "ollama.pid"


def _ollama_started_marker_path() -> Path:
    return _agent_state_dir() / "ollama.started_by_agent_mode"


@contextlib.contextmanager
def _ollama_lock() -> "contextlib.AbstractContextManager[None]":
    state_dir = _agent_state_dir()
    state_dir.mkdir(parents=True, exist_ok=True)

    if fcntl is None:
        # Best-effort only on platforms without fcntl (e.g., Windows).
        yield None
        return

    lock_path = state_dir / "ollama.lock"
    with open(lock_path, "a+b") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            yield None
        finally:
            try:
                fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not text.isdigit():
        return None
    pid = int(text)
    return pid if pid > 0 else None


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _can_signal(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return False


def ensure_host_ollama_running(debug: bool = False) -> None:
    """Ensure a host Ollama server is reachable at 127.0.0.1:11434.

    If not reachable, tries to start `ollama serve` and records a marker so we
    can later stop it only if we started it.
    """

    with _ollama_lock():
        if host_ollama_reachable():
            return

        if shutil.which("ollama") is None:
            raise RuntimeError(
                "Host Ollama is not reachable and 'ollama' is not on PATH. Install the Ollama CLI on the host."
            )

        state_dir = _agent_state_dir()
        pid_path = _ollama_pid_path()
        marker_path = _ollama_started_marker_path()

        existing_pid = _read_pid(pid_path)
        if existing_pid and _pid_exists(existing_pid):
            if debug:
                print(f"[debug] host ollama pid {existing_pid} exists; waiting for readiness", file=sys.stderr)
        else:
            log_path = state_dir / "ollama.log"
            if debug:
                print("[debug] starting host 'ollama serve'", file=sys.stderr)

            popen_kwargs: dict[str, object] = {
                "stdout": open(log_path, "ab"),
                "stderr": subprocess.STDOUT,
            }
            if os.name != "nt":
                popen_kwargs["start_new_session"] = True
            else:
                # CREATE_NEW_PROCESS_GROUP is best-effort; not all Python builds expose it.
                creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if creationflags:
                    popen_kwargs["creationflags"] = creationflags

            proc = subprocess.Popen(["ollama", "serve"], **popen_kwargs)  # type: ignore[arg-type]
            pid_path.write_text(str(proc.pid), encoding="utf-8")
            marker_path.write_text("1", encoding="utf-8")

        deadline = time.time() + 30.0
        while time.time() < deadline:
            if host_ollama_reachable(timeout=1.0):
                return
            time.sleep(0.5)

        raise RuntimeError("Host Ollama failed to become reachable at http://127.0.0.1:11434 after 30s")


def _find_pids_listening_on_11434() -> list[int]:
    pids: list[int] = []

    # Prefer `ss` (usually available on Linux).
    if shutil.which("ss") is not None:
        try:
            out = subprocess.check_output(["ss", "-ltnp"], text=True, stderr=subprocess.DEVNULL)
        except Exception:
            out = ""
        for line in (out or "").splitlines():
            if ":11434" not in line:
                continue
            for token in line.split():
                if "pid=" in token:
                    try:
                        pid_str = token.split("pid=", 1)[1].split(",", 1)[0].strip(")(")
                        pid = int(pid_str)
                        if pid > 0:
                            pids.append(pid)
                    except Exception:
                        pass

    # Fallback to pgrep for 'ollama serve'
    if not pids and shutil.which("pgrep") is not None:
        try:
            out = subprocess.check_output(["pgrep", "-f", "ollama serve"], text=True, stderr=subprocess.DEVNULL)
            for line in (out or "").splitlines():
                try:
                    pid = int(line.strip())
                    if pid > 0:
                        pids.append(pid)
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback to lsof (accurate without root)
    if not pids and shutil.which("lsof") is not None:
        try:
            out = subprocess.check_output(
                ["lsof", "-nP", "-iTCP:11434", "-sTCP:LISTEN", "-t"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            for line in (out or "").splitlines():
                try:
                    pid = int(line.strip())
                    if pid > 0:
                        pids.append(pid)
                except Exception:
                    pass
        except Exception:
            pass

    # Fallback to fuser
    if not pids and shutil.which("fuser") is not None:
        try:
            out = subprocess.check_output(["fuser", "-n", "tcp", "11434"], text=True, stderr=subprocess.DEVNULL)
            for token in (out or "").replace("/tcp:", " ").split():
                if token.isdigit():
                    pid = int(token)
                    if pid > 0:
                        pids.append(pid)
        except Exception:
            pass

    return sorted(set(pids))


def _count_running_labeled_containers(container_bin: str, container_label: str) -> int | None:
    if shutil.which(container_bin) is None:
        return None
    try:
        out = subprocess.check_output(
            [
                container_bin,
                "ps",
                "--filter",
                f"label={container_label}",
                "--format",
                "{{.ID}}",
            ],
            text=True,
        )
    except Exception:
        return None
    ids = [line.strip() for line in (out or "").splitlines() if line.strip()]
    return len(ids)


def maybe_stop_host_ollama_if_last_container(
    container_bin: str,
    container_label: str,
    *,
    debug: bool = False,
    log_prefix: str = "[ai]",
) -> None:
    """Stop the host Ollama process if no labeled agent containers remain.

    Only stops Ollama if the marker file indicates it was started by this tool.
    """

    with _ollama_lock():
        remaining = _count_running_labeled_containers(container_bin, container_label)
        if remaining is not None and remaining > 0:
            if debug:
                print(
                    f"[debug] leaving host ollama running; {remaining} agent container(s) still running",
                    file=sys.stderr,
                )
            return

        if not host_ollama_reachable(timeout=1.0):
            return

        pid_path = _ollama_pid_path()
        marker_path = _ollama_started_marker_path()

        if not marker_path.exists():
            if debug:
                print("[debug] host ollama was not started by this ai; leaving it running", file=sys.stderr)
            return

        pid = _read_pid(pid_path)
        pids: list[int] = []

        if pid and _pid_exists(pid):
            pids = [pid]
        else:
            pids = _find_pids_listening_on_11434()

        if not pids:
            if debug:
                print("[debug] unable to identify ollama pid(s); leaving it running", file=sys.stderr)
            return

        stoppable = [found_pid for found_pid in pids if _can_signal(found_pid)]
        if not stoppable:
            print(
                f"{log_prefix} warning: host Ollama is running but cannot be stopped (permission denied). "
                "It may be running as a system service under a different user.",
                file=sys.stderr,
            )
            if debug:
                print(f"[debug] identified ollama pid(s) but cannot signal: {pids}", file=sys.stderr)
            return

        if debug:
            print(f"[debug] stopping host ollama pid(s): {stoppable}", file=sys.stderr)

        for target in stoppable:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(target, signal.SIGTERM)
                else:
                    os.kill(target, signal.SIGTERM)
            except Exception:
                try:
                    os.kill(target, signal.SIGTERM)
                except Exception:
                    pass

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if not host_ollama_reachable(timeout=0.5):
                break
            time.sleep(0.2)

        for target in stoppable:
            if _pid_exists(target) and host_ollama_reachable(timeout=0.5):
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(target, signal.SIGKILL)
                    else:
                        os.kill(target, signal.SIGKILL)
                except Exception:
                    try:
                        os.kill(target, signal.SIGKILL)
                    except Exception:
                        pass

        if not host_ollama_reachable(timeout=0.5):
            marker_path.unlink(missing_ok=True)
            pid_path.unlink(missing_ok=True)

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from typing import Optional


LOG_PREFIX = "[ollama]"
_detected_gpu_description: Optional[str] = None


def detect_nvidia_gpus() -> Optional[str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    names = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return ", ".join(names) if names else None


def maybe_configure_gpu_defaults(log_prefix: str = LOG_PREFIX) -> Optional[str]:
    global _detected_gpu_description
    if _detected_gpu_description is not None:
        return _detected_gpu_description or None
    gpu_names = detect_nvidia_gpus()
    _detected_gpu_description = gpu_names or ""
    if not gpu_names:
        return None
    if not os.getenv("OLLAMA_USE_GPU"):
        os.environ["OLLAMA_USE_GPU"] = "1"
    if not os.getenv("OLLAMA_GPU_TYPE"):
        os.environ["OLLAMA_GPU_TYPE"] = "cuda"
    print(f"{log_prefix} Detected NVIDIA GPU(s): {gpu_names}", file=sys.stderr)
    if "4090" in gpu_names:
        print(f"{log_prefix} Prioritizing RTX 4090 acceleration.", file=sys.stderr)
    return gpu_names


def ollama_server_running() -> bool:
    if os.getenv("OLLAMA_HOST"):
        return True
    if not shutil.which("ollama"):
        return False
    try:
        subprocess.run(
            ["ollama", "ps"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def wait_for_ollama_server(timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ollama_server_running():
            return True
        time.sleep(0.5)
    return ollama_server_running()


def ensure_ollama_server(debug: bool = False) -> None:
    if os.getenv("OLLAMA_HOST"):
        return
    if ollama_server_running():
        return
    if debug:
        print("[debug] starting local ollama serve process")
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ollama executable not found; please install Ollama.") from exc
    if not wait_for_ollama_server():
        raise RuntimeError("failed to start local ollama serve; please verify your Ollama installation.")


def prepare_runtime(debug: bool = False, log_prefix: str = LOG_PREFIX) -> None:
    maybe_configure_gpu_defaults(log_prefix=log_prefix)
    ensure_ollama_server(debug=debug)

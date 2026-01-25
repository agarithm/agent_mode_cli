#!/bin/bash
set -e

choose_ollama_host() {
    # Prefer localhost (works when containers are launched with --network host).
    local candidates=(
        "http://127.0.0.1:11434"
        "http://host.docker.internal:11434"
    )

    for c in "${candidates[@]}"; do
        if curl -fsS "${c%/}/api/version" >/dev/null 2>&1; then
            export OLLAMA_HOST="$c"
            return 0
        fi
    done

    # If host.docker.internal doesn't resolve, call that out explicitly.
    if ! getent hosts host.docker.internal >/dev/null 2>&1; then
        echo "[entrypoint] ERROR: Ollama not reachable at localhost, and host.docker.internal does not resolve" >&2
        echo "[entrypoint] If you launch containers via the host 'ai' CLI, it should use --network host." >&2
        echo "[entrypoint] For manual docker runs on Linux, use: --network host" >&2
        echo "[entrypoint] Or alternatively add: --add-host=host.docker.internal:host-gateway" >&2
        exit 1
    fi

    echo "[entrypoint] ERROR: host Ollama not reachable." >&2
    echo "[entrypoint] Tried: http://127.0.0.1:11434 and http://host.docker.internal:11434" >&2
    echo "[entrypoint] Start Ollama on the host (the host 'ai' launcher normally does this automatically)." >&2
    exit 1
}


# Check if we're running as root (during build) or developer (at runtime)
if [ "$(id -u)" -eq 0 ]; then
    # Running as root - just execute the command
    exec "$@"
fi

# Running as developer user - requires host Ollama.
choose_ollama_host

echo "[entrypoint] Ollama reachable at ${OLLAMA_HOST}"

# Verify model is available (best-effort)
if curl -fsS "${OLLAMA_HOST%/}/api/tags" 2>/dev/null | grep -q 'gpt-oss'; then
    echo "[entrypoint] gpt-oss model appears available"
else
    echo "[entrypoint] WARNING: gpt-oss model not found. You may need to pull it manually."
fi

# Execute the provided command or default to bash
if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi

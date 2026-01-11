#!/bin/bash
set -e

# Set Ollama environment variables
export OLLAMA_HOME="${OLLAMA_HOME:-/opt/ollama}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-/opt/ollama/models}"
export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

# Check if we're running as root (during build) or developer (at runtime)
if [ "$(id -u)" -eq 0 ]; then
    # Running as root - just execute the command
    exec "$@"
fi

# Running as developer user - start Ollama if not already running
if ! curl -fsS "${OLLAMA_HOST}/api/version" > /dev/null 2>&1; then
    echo "[entrypoint] Starting Ollama service..."
    
    # Start Ollama service in background
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    OLLAMA_PID=$!
    
    # Wait for Ollama to be ready
    for i in $(seq 1 30); do
        if curl -fsS "${OLLAMA_HOST}/api/version" > /dev/null 2>&1; then
            echo "[entrypoint] Ollama service ready (PID: $OLLAMA_PID)"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "[entrypoint] ERROR: Ollama failed to start after 30 seconds" >&2
            if [ -f /tmp/ollama.log ]; then
                cat /tmp/ollama.log >&2
            fi
            exit 1
        fi
        sleep 1
    done
    
    # Verify model is available
    if ! ollama list | grep -q "gpt-oss:latest"; then
        echo "[entrypoint] WARNING: gpt-oss:latest model not found. You may need to pull it manually."
    else
        echo "[entrypoint] gpt-oss:latest model is ready"
    fi
else
    echo "[entrypoint] Ollama service already running"
fi

# Execute the provided command or default to bash
if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi

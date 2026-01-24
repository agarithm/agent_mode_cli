# syntax=docker/dockerfile:1.6

# Stage 1: Use official Ollama image to pull models
FROM ollama/ollama:0.15.0 AS ollama-builder

# Pull the gpt-oss model in builder stage
RUN bash -c '\
    echo "Starting Ollama server..." && \
    ollama serve > /tmp/ollama.log 2>&1 & \
    pid=$! && \
    echo "Waiting for Ollama to be ready..." && \
    while ! ollama list 2>/dev/null | grep -q NAME; do \
        sleep 1; \
    done && \
    echo "Pulling gpt-oss:latest model..." && \
    ollama pull gpt-oss:latest && \
    echo "Model pull complete. Verifying..." && \
    ollama list && \
    echo "Model files:" && \
    find /root/.ollama -type f | wc -l && \
    du -sh /root/.ollama && \
    echo "Stopping Ollama (pid $pid)..." && \
    kill $pid || true && \
    wait $pid 2>/dev/null || true \
'

# Stage 2: Main runtime image
FROM nvidia/cuda:12.6.3-base-ubuntu24.04

LABEL org.opencontainers.image.title="agent-mode-dev"
LABEL org.opencontainers.image.description="Agent-mode CLI developer workspace"
LABEL org.opencontainers.image.source="https://github.com/???"

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV OLLAMA_HOME=/opt/ollama
ENV OLLAMA_MODELS=/opt/ollama/models
ENV OLLAMA_HOST=http://127.0.0.1:11434

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        build-essential \
        ca-certificates \
        curl \
        fd-find \
        git \
        jq \
        less \
        nano \
        net-tools \
        nodejs \
        npm \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        ripgrep \
        unzip \
        vim \
        wget \
    && rm -rf /var/lib/apt/lists/*

COPY rsc/install_ollama.py /tmp/install_ollama.py

# Install Ollama binary
RUN curl -fsSL https://ollama.com/download/ollama-linux-amd64.tgz -o /tmp/ollama.tgz \
    && python3 /tmp/install_ollama.py \
    && cp -r --no-preserve=mode,ownership /tmp/ollama-unpacked/. /usr/local/ \
    && rm -rf /tmp/ollama.tgz /tmp/ollama-unpacked /tmp/install_ollama.py \
    && chmod +x /usr/local/bin/ollama

ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN if ! getent group "${USER_GID}" >/dev/null; then \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    fi \
    && if id -u "${USERNAME}" >/dev/null 2>&1; then \
        usermod --gid "${USER_GID}" "${USERNAME}"; \
    else \
        uid_args=""; \
        if ! getent passwd "${USER_UID}" >/dev/null; then \
            uid_args="--uid ${USER_UID}"; \
        fi; \
        useradd ${uid_args} --gid "${USER_GID}" -m "${USERNAME}"; \
    fi

# Copy preloaded Ollama models from builder stage
COPY --from=ollama-builder /root/.ollama /opt/ollama

# Set ownership of Ollama data directory for non-root user
RUN chown -R developer:${USER_GID} /opt/ollama && \
    chmod -R ug+rwX /opt/ollama

RUN pip3 install --no-cache-dir poetry uv

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && python3 -m playwright install --with-deps chromium \
    && rm /tmp/requirements.txt

COPY dist/ai /usr/local/bin/ai
RUN chmod +x /usr/local/bin/ai

# Copy entrypoint script that starts Ollama service
COPY rsc/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER developer

WORKDIR /workspace
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PS1="(agent-mode) \\u@\\h: \\w\\$ "
ENV AI_IN_CONTAINER=1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["/bin/bash"]

# syntax=docker/dockerfile:1.6

# Main runtime image
FROM ubuntu:24.04

LABEL org.opencontainers.image.title="agent-mode-dev"
LABEL org.opencontainers.image.description="Agent-mode CLI developer workspace"
LABEL org.opencontainers.image.source="https://github.com/???"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1

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

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional, Sequence

from core.universal_context import ChatMessage, ToolCall

try:  # Optional dependency used when available.
    import tiktoken
except Exception:  # pragma: no cover - optional dependency missing
    tiktoken = None  # type: ignore


_DEFAULT_MODEL_LIMITS = {
    "gpt-5.1-codex": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 64_000,
    "gpt-4.1": 128_000,
    "gpt-4.1-mini": 128_000,
    "gpt-4.1-nano": 64_000,
    "qwen2.5-coder:32b": 32_768,
    "qwen2.5-coder:14b": 32_768,
    "gpt-oss:latest": 8_192,
    "xai/grok-3": 131_072,
}


def _env_override_for_model(model: str) -> Optional[int]:
    """Return an env override for the given model name if present."""

    generic = os.getenv("AI_MODEL_CONTEXT_LIMIT")
    if generic:
        try:
            return max(0, int(generic))
        except ValueError:
            pass

    normalized = "".join(ch if ch.isalnum() else "_" for ch in model).upper()
    for suffix in ("CONTEXT_LIMIT", "MAX_TOKENS", "CONTEXT_TOKENS"):
        override = os.getenv(f"AI_MODEL_{normalized}_{suffix}")
        if override:
            try:
                return max(0, int(override))
            except ValueError:
                continue
    return None


def get_model_context_limit(model: Optional[str]) -> Optional[int]:
    if not model:
        return None
    override = _env_override_for_model(model)
    if override:
        return override
    return _DEFAULT_MODEL_LIMITS.get(model)


@lru_cache(maxsize=64)
def _encoding_for_model(model: Optional[str]):
    if tiktoken is None:
        return None
    base_encoding = "cl100k_base"
    try:
        if model:
            return tiktoken.encoding_for_model(model)
    except Exception:
        pass
    try:
        return tiktoken.get_encoding(base_encoding)
    except Exception:  # pragma: no cover - defensive
        return None


def _rough_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def count_text_tokens(text: str, model: Optional[str]) -> int:
    text = text or ""
    if not text:
        return 0
    encoding = _encoding_for_model(model)
    if encoding is not None:
        try:
            return len(encoding.encode(text))
        except Exception:
            pass
    return _rough_token_count(text)


def _tool_calls_to_text(calls: Sequence[ToolCall]) -> str:
    try:
        payload = [
            {
                "name": call.name,
                "arguments": call.arguments,
                "call_id": call.call_id,
            }
            for call in calls
        ]
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except Exception:
        return ""


def estimate_context_tokens(messages: Sequence[ChatMessage], *, model: Optional[str]) -> int:
    total = 0
    for msg in messages:
        if msg.content:
            total += count_text_tokens(str(msg.content), model)
        if msg.tool_calls:
            total += count_text_tokens(_tool_calls_to_text(msg.tool_calls), model)
    return total

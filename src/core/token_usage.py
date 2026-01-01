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


_CONTEXT_WINDOW_FALLBACK = 4_000


def _resolve_global_context_limit() -> int:
    """Resolve the single shared context limit across all models."""

    for env_var in ("AI_CONTEXT_WINDOW", "AI_CONTEXT_LIMIT", "AI_MODEL_CONTEXT_LIMIT"):
        raw_value = os.getenv(env_var)
        if not raw_value:
            continue
        try:
            return max(0, int(raw_value))
        except ValueError:
            continue
    return _CONTEXT_WINDOW_FALLBACK


_GLOBAL_CONTEXT_LIMIT = _resolve_global_context_limit()


def get_model_context_limit(model: Optional[str]) -> Optional[int]:
    del model  # unused; legacy signature kept for compatibility
    return _GLOBAL_CONTEXT_LIMIT


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

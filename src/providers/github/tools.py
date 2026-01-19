from __future__ import annotations

from typing import Any, Dict, List

from providers.tool_schemas import build_openai_chat_completions_tools


def build_tools(prefix: str) -> List[Dict[str, Any]]:
    """Tools schema for GitHub Models (OpenAI Chat Completions compatible)."""

    return build_openai_chat_completions_tools(prefix)

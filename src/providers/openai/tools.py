from __future__ import annotations

from typing import Any, Dict, List

from providers.tool_schemas import build_openai_responses_tools


def build_tools(prefix: str) -> List[Dict[str, Any]]:
    """Tools schema for OpenAI Responses API."""

    return build_openai_responses_tools(prefix)

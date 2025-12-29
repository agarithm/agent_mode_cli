from __future__ import annotations

from typing import Any, Dict, List

from agent_mode_cli.providers._raw_tools import raw_tool_specs


def build_tools(prefix: str) -> List[Dict[str, Any]]:
    """Tools schema for OpenAI Responses API."""

    tools: List[Dict[str, Any]] = []
    for spec in raw_tool_specs(prefix):
        tools.append(
            {
                "type": "function",
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {}),
            }
        )
    return tools

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def _raw_tool_specs(prefix: str) -> Sequence[Dict[str, Any]]:
    prefix = (prefix or "").strip().upper()
    return [
        {
            "name": "set_debug",
            "description": f"Toggle {prefix}_DEBUG on or off.",
            "parameters": {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "Whether debugging should be enabled.",
                    }
                },
                "required": ["enabled"],
            },
        },
        {
            "name": "set_model",
            "description": f"Change the {prefix}_MODEL used for future API calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Model identifier to use.",
                    }
                },
                "required": ["model"],
            },
        },
        {
            "name": "bash",
            "description": "Execute bash commands in a shell. Can run any bash command or script.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    ]


def build_openai_tools(prefix: str) -> List[Dict[str, Any]]:
    """Tools schema for OpenAI Responses API (am)."""
    tools: List[Dict[str, Any]] = []
    for spec in _raw_tool_specs(prefix):
        tools.append(
            {
                "type": "function",
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {}),
            }
        )
    return tools


def build_ollama_tools(prefix: str) -> List[Dict[str, Any]]:
    """Tools schema for Ollama chat API (om)."""
    tools: List[Dict[str, Any]] = []
    for spec in _raw_tool_specs(prefix):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {}),
                },
            }
        )
    return tools


def iter_tool_names() -> List[str]:
    return [spec["name"] for spec in _raw_tool_specs("X")]

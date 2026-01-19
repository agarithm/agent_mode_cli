from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from providers._raw_tools import raw_tool_specs


def _normalize_spec(spec: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    name = str(spec.get("name") or "").strip()
    description = str(spec.get("description") or "")
    parameters = spec.get("parameters") or {}
    if not isinstance(parameters, dict):
        parameters = {}
    return name, description, parameters


def build_openai_responses_tools(prefix: str) -> List[Dict[str, Any]]:
    """Build tool schema compatible with the OpenAI Responses API.

    Shape matches what `src/providers/openai/tools.py` historically produced.
    """

    tools: List[Dict[str, Any]] = []
    for spec in raw_tool_specs(prefix):
        name, description, parameters = _normalize_spec(spec)
        tools.append(
            {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
            }
        )
    return tools


def build_openai_chat_completions_tools(prefix: str) -> List[Dict[str, Any]]:
    """Build tool schema compatible with OpenAI Chat Completions style APIs.

    Used by GitHub Models and Ollama chat tool calling.
    Shape matches what `src/providers/github/tools.py` and `src/providers/ollama/tools.py`
    historically produced.
    """

    tools: List[Dict[str, Any]] = []
    for spec in raw_tool_specs(prefix):
        name, description, parameters = _normalize_spec(spec)
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return tools

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence


def _raw_tool_specs(prefix: str) -> Sequence[Dict[str, Any]]:
    prefix = (prefix or "").strip().upper()
    return [
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
        {
            "name": "web_fetch",
            "description": "Fetch a URL over HTTP(S) (no JavaScript). Follows redirects and can extract readable text from HTML.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (http or https).",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Request timeout in seconds (default: 20).",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "Maximum response bytes to read (default: 1500000).",
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "If true, converts HTML into readable text (default: true).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return in the body (default: 20000).",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as a JSON object.",
                        "additionalProperties": True,
                    },
                },
                "required": ["url"],
            },
        },
        {
            "name": "js_web_fetch",
            "description": "Fetch a URL using a real headless browser with JavaScript enabled (Playwright/Chromium). Use when web_fetch returns insufficient content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (http or https).",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Navigation timeout in seconds (default: 30).",
                    },
                    "wait_until": {
                        "type": "string",
                        "description": "When to consider navigation finished: load | domcontentloaded | networkidle | commit (default: networkidle).",
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "If true, returns rendered body text; if false, returns rendered HTML (default: true).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return in the body (default: 20000).",
                    },
                    "user_agent": {
                        "type": "string",
                        "description": "Optional user agent override.",
                    },
                },
                "required": ["url"],
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

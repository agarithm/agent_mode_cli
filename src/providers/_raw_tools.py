from __future__ import annotations

from typing import Any, Dict, List, Sequence


def raw_tool_specs(prefix: str) -> Sequence[Dict[str, Any]]:
    """Return the provider-agnostic raw tool specs.

    Providers can map these into their own tool schema shape.
    """

    prefix = (prefix or "").strip().upper()
    return [
        {
            "name": "list_dir",
            "description": "Safely list a directory within the current working directory (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: '.'). Must stay within the workspace.",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "If true, list recursively (default: false).",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum recursion depth when recursive=true (default: 2).",
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum number of entries to return (default: 2000).",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "description": "If true, include perms/size/mtime columns (tab-separated) for each entry (default: false).",
                    },
                },
            },
        },
        {
            "name": "read_file",
            "description": "Safely read a file within the current working directory (read-only). Supports byte offsets and bounded reads.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path to read. Must stay within the workspace.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Byte offset to start reading from (default: 0).",
                    },
                    "length": {
                        "type": "integer",
                        "description": "Maximum bytes to read (optional). If omitted, reads up to a fixed safe maximum.",
                    },
                },
                "required": ["path"],
            },
        },
        {
            "name": "python_exec",
            "description": "Execute a short Python snippet in a subprocess and return a JSON result (captures stdout/stderr). This tool can execute arbitrary code and should be used carefully.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. To return a value, assign it to a variable named 'result'.",
                    },
                    "input": {
                        "type": "object",
                        "description": "Optional JSON object provided to the snippet as 'tool_input'.",
                        "additionalProperties": True,
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 10).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters returned for stdout/stderr/traceback (default: 20000).",
                    },
                },
                "required": ["code"],
            },
        },
        {
            "name": "search_files",
            "description": "Search for text in files under the current working directory using ripgrep (rg). Read-only, no shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (regex by default unless is_regex=false).",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of relative paths to search under (default: ['.']).",
                    },
                    "is_regex": {
                        "type": "boolean",
                        "description": "If true, treat query as a regex; if false, search as a fixed string (default: true).",
                    },
                    "ignore_case": {
                        "type": "boolean",
                        "description": "If true, case-insensitive search (default: false).",
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional rg --glob filter (e.g., '*.py' or 'src/**').",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of context lines before/after each match (default: 0).",
                    },
                    "max_matches": {
                        "type": "integer",
                        "description": "Maximum matches per file (default: 200).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default: 40000).",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "edit_file",
            "description": "Edit a file within the current working directory. Supports structured edits and whole-file overwrite/append. This tool modifies files and should be used carefully.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path (workspace-confined).",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Write mode: overwrite | append | edits. overwrite/append use 'content'; edits uses the 'edits' list.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content used when mode is set (overwrite replaces entire file; append adds to end).",
                    },
                    "edits": {
                        "type": "array",
                        "description": "List of edit operations (objects) to apply in order.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {
                                    "type": "string",
                                    "description": "Operation: replace | delete | insert_before | insert_after",
                                },
                                "old": {"type": "string"},
                                "new": {"type": "string"},
                                "before": {"type": "string"},
                                "after": {"type": "string"},
                                "content": {"type": "string"},
                                "count": {
                                    "type": "integer",
                                    "description": "How many occurrences to apply (0 = all).",
                                },
                            },
                            "required": ["op"],
                        },
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, do not write; return a unified diff only (default: false).",
                    },
                    "make_backup": {
                        "type": "boolean",
                        "description": "If true and file exists, create a .bak.* backup before writing (default: true).",
                    },
                },
                "required": ["path"],
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
        {
            "name": "http_fetch",
            "description": "Fetch a URL over HTTP(S). mode='simple' uses HTTP only (no JS). mode='browser' uses Playwright/Chromium with JavaScript enabled.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (http or https).",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Fetch mode: simple | browser (default: simple).",
                    },
                    "timeout_seconds": {
                        "type": "integer",
                        "description": "Timeout in seconds (default: 20 for simple, 30 for browser).",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": "(simple mode) Maximum response bytes to read (default: 1500000).",
                    },
                    "extract_text": {
                        "type": "boolean",
                        "description": "If true, returns readable text; otherwise returns HTML (default: true).",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return in the body (default: 20000).",
                    },
                    "headers": {
                        "type": "object",
                        "description": "(simple mode) Optional HTTP headers as a JSON object.",
                        "additionalProperties": True,
                    },
                    "wait_until": {
                        "type": "string",
                        "description": "(browser mode) When to consider navigation finished: load | domcontentloaded | networkidle | commit (default: networkidle).",
                    },
                    "user_agent": {
                        "type": "string",
                        "description": "(browser mode) Optional user agent override.",
                    },
                },
                "required": ["url"],
            },
        },
    ]


def iter_tool_names() -> List[str]:
    return [spec["name"] for spec in raw_tool_specs("X")]

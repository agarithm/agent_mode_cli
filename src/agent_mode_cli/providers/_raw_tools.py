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
            "name": "file_metadata",
            "description": "Return basic metadata about a file or directory within the current working directory (read-only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to file or directory (workspace-confined).",
                    }
                },
                "required": ["path"],
            },
        },
        {
            "name": "git_status",
            "description": "Read-only git status for the workspace repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default: 40000).",
                    }
                },
            },
        },
        {
            "name": "git_diff",
            "description": "Read-only git diff for the workspace repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "description": "Optional path or list of paths to limit diff to (workspace-confined).",
                        "oneOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return (default: 40000).",
                    },
                    "staged": {
                        "type": "boolean",
                        "description": "If true, returns staged diff (git diff --staged) (default: false).",
                    },
                    "include_untracked": {
                        "type": "boolean",
                        "description": "If true, includes a diff-style preview for untracked text files (default: true).",
                    },
                    "max_untracked_files": {
                        "type": "integer",
                        "description": "Maximum untracked files to include when include_untracked=true (default: 20).",
                    },
                    "max_untracked_file_bytes": {
                        "type": "integer",
                        "description": "Maximum bytes read per untracked file when include_untracked=true (default: 200000).",
                    },
                },
            },
        },
        {
            "name": "write_file",
            "description": "Write content to a file within the current working directory. This tool modifies files and should be used carefully.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path (workspace-confined).",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write.",
                    },
                    "mode": {
                        "type": "string",
                        "description": "Write mode: overwrite | append | insert (default: overwrite).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Byte offset for mode=insert (default: 0).",
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
                "required": ["path", "content"],
            },
        },
        {
            "name": "edit_file",
            "description": "Apply structured edits (replace/insert/delete) to a file within the current working directory. This tool modifies files and should be used carefully.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Target file path (workspace-confined).",
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
                "required": ["path", "edits"],
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


def iter_tool_names() -> List[str]:
    return [spec["name"] for spec in raw_tool_specs("X")]

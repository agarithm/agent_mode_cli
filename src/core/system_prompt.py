from __future__ import annotations

from typing import Iterable

from core.confirm import requires_confirmation
from providers._raw_tools import raw_tool_specs


_READ_ONLY_TOOLS = {"list_dir", "read_file", "search_files", "http_fetch"}


def _format_args_summary(required: Iterable[str], optional: Iterable[str], *, max_optional: int = 4) -> str:
    required_list = [r for r in required if (r or "").strip()]
    optional_list = [o for o in optional if (o or "").strip()]

    parts: list[str] = []
    if required_list:
        parts.append("required: " + ", ".join(required_list))
    if optional_list:
        shown = optional_list[:max_optional]
        suffix = "" if len(optional_list) <= max_optional else ", …"
        parts.append("optional: " + ", ".join(shown) + suffix)
    return "; ".join(parts) if parts else "no args"


def _build_tools_section(prefix: str) -> str:
    lines: list[str] = ["Tools available:"]
    for spec in raw_tool_specs(prefix):
        name = (spec.get("name") or "").strip()
        if not name:
            continue

        params = spec.get("parameters") or {}
        properties = params.get("properties") or {}
        required = params.get("required") or []
        optional = [k for k in properties.keys() if k not in set(required)]

        kind = "read-only" if name in _READ_ONLY_TOOLS else "mutating"
        confirm = "; confirm" if requires_confirmation(name) else ""

        desc = (spec.get("description") or "").strip()
        if len(desc) > 160:
            desc = desc[:160].rstrip() + "…"

        args = _format_args_summary(required, optional)
        lines.append(f"- {name} ({args}) [{kind}{confirm}] — {desc}")
    return "\n".join(lines)


def build_internal_system_prompt(agent_name: str) -> str:
    agent_name = (agent_name or "sloppy").strip() or "AGENT"
    tools_section = _build_tools_section("AI")
    return "\n".join(
        [
            f"You are {agent_name}, a self-aware coding agent running in a terminal.",
            tools_section,
            "Guidelines:",
            "- Prefer read-only tools (list_dir/read_file/search_files/http_fetch) for inspection.",
            "- Use python_exec for small computations; use bash for normal shell tasks; use edit_file for file changes.",
            "- Some mutating tools may require interactive user confirmation before execution.",
            "- Stop and ask for help if you are confused or stuck. Don't guess.",
            "- Confine your edits to the current working directory and its subdirectories.",
        ]
    )

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _normalize_bool(value: str) -> Optional[bool]:
    v = (value or "").strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}:
        return True
    if v in {"0", "false", "f", "no", "n", "off", "disable", "disabled"}:
        return False
    return None


def _normalize_optional_float(value: str) -> Optional[float]:
    v = (value or "").strip().lower()
    if not v:
        return None
    if v in {"off", "none", "null", "0", "0.0"}:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _normalize_positive_int(value: str) -> Optional[int]:
    v = (value or "").strip()
    if not v:
        return None
    try:
        parsed = int(v, 10)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


@dataclass
class RuntimeSettings:
    debug: bool
    max_tool_iterations: int
    max_tool_seconds: Optional[float]

    def format(self) -> str:
        mt = self.max_tool_iterations
        ms = self.max_tool_seconds
        ms_text = "off" if ms is None else f"{ms:.1f}s"
        return "\n".join(
            [
                "Settings:",
                f"- debug: {'on' if self.debug else 'off'}",
                f"- max_tool_iterations: {mt}",
                f"- max_tool_seconds: {ms_text}",
            ]
        )

    def set_debug_from_text(self, value: str) -> str:
        parsed = _normalize_bool(value)
        if parsed is None:
            return "error: debug must be one of: on/off, true/false, 1/0"
        self.debug = parsed
        return f"debug {'enabled' if self.debug else 'disabled'}."

    def set_max_tool_iterations_from_text(self, value: str) -> str:
        parsed = _normalize_positive_int(value)
        if parsed is None:
            return "error: max_tool_iterations must be a positive integer"
        self.max_tool_iterations = parsed
        return f"max_tool_iterations set to {self.max_tool_iterations}."

    def set_max_tool_seconds_from_text(self, value: str) -> str:
        parsed = _normalize_optional_float(value)
        if parsed is not None and parsed <= 0:
            return "error: max_tool_seconds must be > 0, or 'off'"
        self.max_tool_seconds = parsed
        if self.max_tool_seconds is None:
            return "max_tool_seconds disabled."
        return f"max_tool_seconds set to {self.max_tool_seconds:.1f}s."

from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence

from agent_mode_cli.core.agent_loop import ParseResult
from agent_mode_cli.core.universal_context import UniversalContext


class ProviderAdapter(Protocol):
    """Contract for provider-specific model adapters used by the agent runner."""

    def call_model(self, *, model: str, tools: Sequence[Dict[str, Any]], context: UniversalContext, debug: bool) -> Any:
        ...

    def parse_response(self, response: Any, *, debug: bool) -> ParseResult:
        ...

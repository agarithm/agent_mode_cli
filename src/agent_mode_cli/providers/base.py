from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Sequence

from agent_mode_cli.core.agent_loop import ParseResult
from agent_mode_cli.core.universal_context import UniversalContext


class ProviderRateLimitError(RuntimeError):
    """Exception raised when a provider signals a transient rate-limit."""

    def __init__(self, message: str, *, provider: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.retry_after = retry_after


class ProviderAdapter(Protocol):
    """Contract for provider-specific model adapters used by the agent runner."""

    def list_models(self, *, debug: bool) -> Sequence[str]:
        ...

    def call_model(self, *, model: str, tools: Sequence[Dict[str, Any]], context: UniversalContext, debug: bool) -> Any:
        ...

    def parse_response(self, response: Any, *, debug: bool) -> ParseResult:
        ...

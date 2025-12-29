"""Provider implementations and provider-facing contracts.

Adding a new provider should generally mean:
- Create a new subpackage under this package (e.g. `providers/foo/`).
- Implement an adapter that satisfies `ProviderAdapter`.
- Optionally provide runtime prep helpers and provider-specific tool schema builders.
"""

from agent_mode_cli.providers.base import ProviderAdapter

__all__ = ["ProviderAdapter"]

"""Provider implementations and provider-facing contracts.

Adding a new provider should generally mean:
- Create a new subpackage under this package (e.g. `providers/foo/`).
- Implement an adapter that satisfies `ProviderAdapter`.
- Optionally provide runtime prep helpers and provider-specific tool schema builders.
"""

from providers.base import ProviderAdapter, ProviderRateLimitError

__all__ = ["ProviderAdapter", "ProviderRateLimitError"]

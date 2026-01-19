"""GitHub Models provider implementation.

This provider is backed by the GitHub Models inference REST API:
https://models.github.ai/inference/chat/completions
"""

from providers.github.adapter import GitHubProviderAdapter
from providers.github.runtime import create_github_models_client
from providers.github.tools import build_tools

__all__ = ["GitHubProviderAdapter", "create_github_models_client", "build_tools"]

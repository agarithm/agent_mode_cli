"""GitHub Copilot / GitHub Models provider implementation.

This provider is backed by the GitHub Models inference REST API:
https://models.github.ai/inference/chat/completions
"""

from providers.copilot.adapter import CopilotProviderAdapter
from providers.copilot.runtime import create_github_models_client
from providers.copilot.tools import build_tools

__all__ = ["CopilotProviderAdapter", "create_github_models_client", "build_tools"]

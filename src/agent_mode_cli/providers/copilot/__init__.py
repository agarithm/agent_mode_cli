"""GitHub Copilot / GitHub Models provider implementation.

This provider is backed by the GitHub Models inference REST API:
https://models.github.ai/inference/chat/completions
"""

from agent_mode_cli.providers.copilot.adapter import CopilotProviderAdapter
from agent_mode_cli.providers.copilot.runtime import create_github_models_client
from agent_mode_cli.providers.copilot.tools import build_tools

__all__ = ["CopilotProviderAdapter", "create_github_models_client", "build_tools"]

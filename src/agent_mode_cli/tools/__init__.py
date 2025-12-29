"""Tool implementations for agent_mode_cli.

Each tool should expose a callable with the same signature used by the agent
runner, and be re-exported here for convenience.
"""

from .bash import bash_command
from .web_fetch import web_fetch
from .js_web_fetch import js_web_fetch

__all__ = ["bash_command", "web_fetch", "js_web_fetch"]

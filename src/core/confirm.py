from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Mapping, Optional


def _in_container() -> bool:
    """Check if running inside a container environment."""
    return os.getenv("AI_IN_CONTAINER", "").lower() in ("1", "true", "yes", "on")


@dataclass
class ConfirmState:
    approve_all: bool = False
    debug: bool = False
    _pre_approve_debug: Optional[bool] = None

    def enable_approve_all(self) -> None:
        if self.approve_all:
            return
        self.approve_all = True
        self._pre_approve_debug = self.debug
        if not self.debug:
            self.debug = True
        print("[confirm] All subsequent confirmations auto-approved for this user prompt.")

    def reset_after_prompt(self) -> None:
        if not self.approve_all:
            self._pre_approve_debug = None
            return
        if self._pre_approve_debug is not None and self.debug != self._pre_approve_debug:
            self.debug = self._pre_approve_debug
        self.approve_all = False
        self._pre_approve_debug = None


def requires_confirmation(tool_name: Optional[str]) -> bool:
    return tool_name in {"bash", "edit_file", "python_exec"}


def prompt_for_confirmation(tool_name: str, arguments: Mapping[str, Any], state: ConfirmState) -> bool:
    # Skip confirmations when running inside a container
    if _in_container():
        if state.debug:
            print(f"[debug] auto-approved '{tool_name}' (running in container)", file=sys.stderr)
        return True
    
    if state.approve_all:
        if state.debug:
            if tool_name == "bash":
                cmd = None
                try:
                    cmd = arguments.get("command")  # type: ignore[attr-defined]
                except Exception:
                    cmd = None
                if isinstance(cmd, str) and cmd.strip():
                    print(f"[debug] auto-approved 'bash' (approve-all): {cmd}", file=sys.stderr)
                else:
                    print("[debug] auto-approved 'bash' (approve-all)", file=sys.stderr)
            else:
                print(f"[debug] auto-approved '{tool_name}' due to approve-all mode", file=sys.stderr)
                if arguments:
                    try:
                        pretty_args = json.dumps(dict(arguments), indent=2)
                    except TypeError:
                        pretty_args = str(arguments)
                    print(f"[debug] arguments:\n{pretty_args}", file=sys.stderr)
        return True
    print(f"[confirm] The '{tool_name}' tool may modify files or system state.")
    if arguments:
        try:
            pretty_args = json.dumps(dict(arguments), indent=2)
        except TypeError:
            pretty_args = str(arguments)
        print(f"[confirm] Arguments:\n{pretty_args}")
    while True:
        try:
            answer = input("Proceed? [y/N/a]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            if state.debug:
                print("[debug] confirmation prompt interrupted; defaulting to no")
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("a", "all"):
            state.enable_approve_all()
            return True
        if answer in ("", "n", "no"):
            return False
        print("Please respond with 'y' or 'n'.")

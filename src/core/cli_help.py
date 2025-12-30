from __future__ import annotations

import sys
from typing import Iterable, Optional


def print_version(version: str) -> None:
    print(version)


def print_help(
    *,
    usage: str,
    description: Optional[str] = None,
    env_lines: Optional[Iterable[str]] = None,
) -> None:
    print(f"usage: {usage}")
    if description:
        print(f"\n{description}")
    if env_lines:
        print("\nenv:")
        for line in env_lines:
            print(f"  {line}")


def handle_common_flags(
    argv: list[str],
    *,
    usage: str,
    description: Optional[str] = None,
    env_lines: Optional[Iterable[str]] = None,
    version: Optional[str] = None,
) -> Optional[int]:
    if argv and argv[0] in ("-h", "--help"):
        print_help(usage=usage, description=description, env_lines=env_lines)
        return 0
    if argv and argv[0] == "--version":
        if version is None:
            print("unknown", file=sys.stderr)
            return 1
        print_version(version)
        return 0
    return None

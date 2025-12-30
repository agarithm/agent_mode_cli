from __future__ import annotations

import os
from typing import Optional


def load_user_prompt(env_var: str, default_filename: str, debug: bool = False) -> Optional[str]:
    prompt_file = os.getenv(env_var) or os.path.join(os.path.expanduser("~"), default_filename)
    try:
        if prompt_file and os.path.isfile(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as handle:
                content = handle.read().strip()
                return content or None
    except Exception as exc:
        if debug:
            print(f"[debug] failed to read prompt file {prompt_file}: {exc}")
    return None

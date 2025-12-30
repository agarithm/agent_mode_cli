from __future__ import annotations

from typing import Any, Mapping, Optional

from .js_web_fetch import js_web_fetch
from .web_fetch import web_fetch


_DEFAULT_SIMPLE_TIMEOUT_SECONDS = 20
_DEFAULT_BROWSER_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_BYTES = 1_500_000
_DEFAULT_MAX_CHARS = 20_000


def http_fetch(
    url: str = "",
    *,
    mode: str = "simple",
    timeout_seconds: Optional[int] = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    extract_text: bool = True,
    max_chars: int = _DEFAULT_MAX_CHARS,
    headers: Optional[Mapping[str, Any]] = None,
    wait_until: str = "networkidle",
    user_agent: Optional[str] = None,
) -> str:
    """Fetch a URL over HTTP(S).

    Modes:
    - simple: HTTP(S) fetch without JavaScript
    - browser: JavaScript-enabled fetch using Playwright/Chromium

    This is a thin wrapper that consolidates the underlying implementations into one tool.
    """

    url = (url or "").strip()
    if not url:
        return "error: url is required"

    mode = (mode or "").strip().lower() or "simple"
    if mode not in {"simple", "browser"}:
        return "error: mode must be one of: simple, browser"

    if max_chars <= 0:
        return "error: max_chars must be > 0"

    if timeout_seconds is None:
        timeout_seconds = (
            _DEFAULT_BROWSER_TIMEOUT_SECONDS if mode == "browser" else _DEFAULT_SIMPLE_TIMEOUT_SECONDS
        )

    if timeout_seconds <= 0:
        return "error: timeout_seconds must be > 0"

    if mode == "browser":
        return js_web_fetch(
            url,
            timeout_seconds=int(timeout_seconds),
            wait_until=wait_until,
            extract_text=extract_text,
            max_chars=max_chars,
            user_agent=user_agent,
        )

    if max_bytes <= 0:
        return "error: max_bytes must be > 0"

    return web_fetch(
        url,
        timeout_seconds=int(timeout_seconds),
        max_bytes=max_bytes,
        extract_text=extract_text,
        max_chars=max_chars,
        headers=headers,
    )

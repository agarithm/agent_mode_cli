from __future__ import annotations

import json
from typing import Any, Mapping, Optional


_DEFAULT_TIMEOUT_SECONDS = 30
_DEFAULT_MAX_CHARS = 20_000


def js_web_fetch(
    url: str = "",
    *,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    wait_until: str = "networkidle",
    extract_text: bool = True,
    max_chars: int = _DEFAULT_MAX_CHARS,
    user_agent: Optional[str] = None,
) -> str:
    """Fetch a web URL using a real headless browser (JS enabled).

    Backed by Playwright (Chromium). This is intended as a fallback for websites
    that require JavaScript rendering and return insufficient HTML to basic
    HTTP fetchers.
    """

    url = (url or "").strip()
    if not url:
        return "error: url is required"

    if timeout_seconds <= 0:
        return "error: timeout_seconds must be > 0"
    if max_chars <= 0:
        return "error: max_chars must be > 0"

    wait_until = (wait_until or "").strip().lower() or "networkidle"
    if wait_until not in {"load", "domcontentloaded", "networkidle", "commit"}:
        return "error: wait_until must be one of: load, domcontentloaded, networkidle, commit"

    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return (
            "error: playwright is not installed\n"
            "To enable js_web_fetch:\n"
            "  pip install playwright\n"
            "  python -m playwright install chromium"
        )

    timeout_ms = int(timeout_seconds * 1000)
    ua = (user_agent or "").strip() or "agent_mode_cli/1.0 (js_web_fetch; Playwright)"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = None
            try:
                context = browser.new_context(user_agent=ua)
                page = context.new_page()

                response = page.goto(url, wait_until=wait_until, timeout=timeout_ms)

                # Some JS-heavy sites keep network busy; also wait for DOM to stabilize a bit.
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
                except Exception:
                    pass

                status = getattr(response, "status", None)
                final_url = page.url

                if extract_text:
                    try:
                        body = page.inner_text("body")
                    except Exception:
                        body = page.content()
                else:
                    body = page.content()

                body = (body or "").strip()
                truncated = False
                if len(body) > max_chars:
                    body = body[:max_chars]
                    truncated = True

                meta = {
                    "status": int(status) if isinstance(status, int) else status,
                    "url": final_url,
                    "wait_until": wait_until,
                }
                header = "\n".join([
                    f"status: {meta['status']}",
                    f"url: {meta['url']}",
                    f"wait_until: {meta['wait_until']}",
                ])
                if truncated:
                    header += f"\nnote: truncated to max_chars={max_chars}"

                return header + "\n---\n" + (body or "(no content)")
            finally:
                if context is not None:
                    try:
                        context.close()
                    except Exception:
                        pass
                try:
                    browser.close()
                except Exception:
                    pass

    except Exception as exc:
        # Provide a helpful hint for the common missing-browser case.
        message = str(exc) or repr(exc)
        lowered = message.lower()
        if "executable doesn't exist" in lowered or "browser type" in lowered or "playwright install" in lowered:
            return (
                "error: Playwright browser not installed\n"
                "Run:\n"
                "  python -m playwright install chromium\n"
                f"Details: {message}"
            )
        return "error: " + json.dumps({"message": message}, indent=2)

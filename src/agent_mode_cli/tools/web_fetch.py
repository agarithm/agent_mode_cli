from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

import httpx
from bs4 import BeautifulSoup


_DEFAULT_MAX_BYTES = 1_500_000
_DEFAULT_MAX_CHARS = 20_000
_DEFAULT_TIMEOUT_SECONDS = 20


def _normalize_headers(headers: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    if not headers:
        return {}
    out: Dict[str, str] = {}
    for key, value in dict(headers).items():
        if value is None:
            continue
        out[str(key)] = str(value)
    return out


def _html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements.
    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            try:
                tag.extract()
            except Exception:
                pass

    text = soup.get_text("\n", strip=True)
    # Compress excessive blank lines.
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    last_blank = False
    for line in lines:
        if not line:
            if last_blank:
                continue
            last_blank = True
            cleaned.append("")
            continue
        last_blank = False
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def web_fetch(
    url: str = "",
    *,
    timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    extract_text: bool = True,
    max_chars: int = _DEFAULT_MAX_CHARS,
    headers: Optional[Mapping[str, Any]] = None,
) -> str:
    """Fetch a URL over HTTP(S) and return content.

    This is a lightweight, non-JS fetch designed to be safer and more reliable
    than shelling out to curl. It follows redirects and optionally extracts
    readable text from HTML.
    """

    url = (url or "").strip()
    if not url:
        return "error: url is required"

    if max_bytes <= 0:
        return "error: max_bytes must be > 0"
    if max_chars <= 0:
        return "error: max_chars must be > 0"
    if timeout_seconds <= 0:
        return "error: timeout_seconds must be > 0"

    req_headers = {
        "User-Agent": "agent_mode_cli/1.0 (+https://github.com)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.5",
    }
    req_headers.update(_normalize_headers(headers))

    try:
        timeout = httpx.Timeout(timeout_seconds)
        with httpx.Client(follow_redirects=True, timeout=timeout, headers=req_headers) as client:
            resp = client.get(url)

        status = resp.status_code
        final_url = str(resp.url)
        content_type = (resp.headers.get("content-type") or "").strip()

        raw = resp.content or b""
        truncated_bytes = False
        if len(raw) > max_bytes:
            raw = raw[:max_bytes]
            truncated_bytes = True

        # Decide how to decode.
        text: str
        if "text/" in content_type or "json" in content_type or not content_type:
            text = resp.text
            if truncated_bytes:
                # httpx resp.text uses the full body; re-decode from truncated bytes.
                encoding = resp.encoding or "utf-8"
                text = raw.decode(encoding, errors="replace")
        else:
            # Binary content is out of scope for a lightweight text fetch tool.
            meta = {
                "status": status,
                "url": final_url,
                "content_type": content_type,
                "bytes": len(resp.content or b""),
            }
            return "error: non-text content type\n" + json.dumps(meta, indent=2)

        body: str
        if extract_text and ("text/html" in content_type or "application/xhtml+xml" in content_type or "<html" in text.lower()):
            body = _html_to_text(text)
        else:
            body = text.strip()

        truncated_chars = False
        if len(body) > max_chars:
            body = body[:max_chars]
            truncated_chars = True

        header_lines = [
            f"status: {status}",
            f"url: {final_url}",
        ]
        if content_type:
            header_lines.append(f"content_type: {content_type}")
        header_lines.append(f"bytes: {len(resp.content or b'')}")
        if truncated_bytes:
            header_lines.append(f"note: truncated to max_bytes={max_bytes}")
        if truncated_chars:
            header_lines.append(f"note: truncated to max_chars={max_chars}")

        return "\n".join(header_lines) + "\n---\n" + (body or "(no content)")

    except httpx.TimeoutException:
        return f"error: request timed out after {timeout_seconds} seconds"
    except httpx.HTTPError as exc:
        return f"error: http error - {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return f"error: {exc}"

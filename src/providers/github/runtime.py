from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional

import httpx


DEFAULT_BASE_URL = "https://models.github.ai"
DEFAULT_API_VERSION = "2022-11-28"

# A pseudo model ID that we resolve to the latest available Gemini model.
GEMINI_LATEST_ALIASES = {"gemini-latest", "google/gemini-latest", "gemini"}

# If we cannot reach the catalog, fall back to a model ID that is widely shown in
# GitHub Models docs/examples (and is likely to exist).
CATALOG_FALLBACK_MODEL = "openai/gpt-4.1"


def get_github_models_token() -> str:
    """Return the token used for GitHub Models inference.

    The GitHub Models API uses a GitHub token (PAT / GitHub App token / Actions token)
    with `models: read` permissions.
    """

    for key in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT", "CM_TOKEN"):
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    raise RuntimeError(
        "Missing GitHub token for GitHub Models. Set GITHUB_TOKEN (recommended) or GH_TOKEN. "
        "Token must have models access (fine-grained: Models read; classic: models scope)."
    )


def create_github_models_client(
    *,
    token: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    api_version: str = DEFAULT_API_VERSION,
    timeout_seconds: float = 60.0,
) -> httpx.Client:
    token = (token or "").strip() or get_github_models_token()

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": api_version,
    }

    return httpx.Client(base_url=base_url, headers=headers, timeout=timeout_seconds)


_resolved_gemini_model: Optional[str] = None


def _looks_like_gemini(model_id: str, *, name: str, publisher: str) -> bool:
    mid = (model_id or "").strip().lower()
    nm = (name or "").strip().lower()
    pub = (publisher or "").strip().lower()
    if ("gemini" in mid) and (mid.startswith("google/") or mid.startswith("google-ai/") or mid.startswith("googleai/")):
        return True
    if "gemini" in nm and ("google" in pub or mid.startswith("google/")):
        return True
    return False


def _choose_latest_model(models: Iterable[Dict[str, Any]]) -> Optional[str]:
    # Prefer models that explicitly support tool-calling.
    candidates: List[Dict[str, Any]] = []
    for m in models:
        mid = str(m.get("id") or "").strip()
        if not mid:
            continue
        caps = m.get("capabilities") or []
        if isinstance(caps, list) and "tool-calling" not in {str(c).strip().lower() for c in caps}:
            continue
        candidates.append(m)

    if not candidates:
        candidates = list(models)

    def key(m: Dict[str, Any]) -> tuple[str, str]:
        version = str(m.get("version") or "").strip()  # often YYYY-MM-DD
        mid = str(m.get("id") or "").strip()
        return (version, mid)

    best = None
    for m in candidates:
        if best is None or key(m) > key(best):
            best = m
    if not best:
        return None
    mid = str(best.get("id") or "").strip()
    return mid or None


def resolve_gemini_model(client: httpx.Client) -> str:
    """Resolve the latest Gemini model ID from the GitHub Models catalog.

    Uses the Models catalog endpoint and caches the result for the process.
    Falls back to `CATALOG_FALLBACK_MODEL` if the catalog is unavailable.
    """

    global _resolved_gemini_model
    if _resolved_gemini_model:
        return _resolved_gemini_model

    try:
        resp = client.get("/catalog/models")
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            raise RuntimeError("unexpected catalog response")

        gemini_models = [
            m
            for m in data
            if isinstance(m, dict)
            and _looks_like_gemini(
                str(m.get("id") or ""),
                name=str(m.get("name") or ""),
                publisher=str(m.get("publisher") or ""),
            )
        ]
        chosen = _choose_latest_model(gemini_models)
        if chosen:
            _resolved_gemini_model = chosen
            return chosen

        # If we couldn't positively identify Gemini models, pick the newest tool-calling model.
        chosen_any = _choose_latest_model([m for m in data if isinstance(m, dict)])
        _resolved_gemini_model = chosen_any or CATALOG_FALLBACK_MODEL
        return _resolved_gemini_model
    except Exception:
        _resolved_gemini_model = CATALOG_FALLBACK_MODEL
        return _resolved_gemini_model


def maybe_resolve_model_alias(model: str, client: httpx.Client) -> str:
    m = (model or "").strip()
    if not m:
        return resolve_gemini_model(client)
    if m.strip().lower() in GEMINI_LATEST_ALIASES:
        return resolve_gemini_model(client)
    return m

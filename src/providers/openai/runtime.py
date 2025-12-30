from __future__ import annotations

from openai import OpenAI

try:  # openai>=1.x
    from openai import OpenAIError  # type: ignore
except Exception:  # pragma: no cover - defensive
    OpenAIError = Exception  # type: ignore


def create_openai_client() -> OpenAI:
    try:
        return OpenAI()
    except OpenAIError as exc:
        message = (str(exc) or "").strip() or "OpenAI client initialization failed"
        if "api_key" in message.lower() or "OPENAI_API_KEY" in message:
            raise RuntimeError("OPENAI_API_KEY is not set") from exc
        raise RuntimeError(message) from exc

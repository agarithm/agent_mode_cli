from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Sequence

import ollama

from providers.ollama.adapter import OllamaProviderAdapter
from providers.ollama.runtime import prepare_runtime


def _pull_ollama_model(adapter: OllamaProviderAdapter, model: str) -> None:
    print(f"[ollama] Pulling model '{model}' (this may take a while)...", file=sys.stderr)
    try:
        result = adapter.client.pull(model=model, stream=True)
    except TypeError:
        result = adapter.client.pull(model=model)
    except Exception as exc:  # pragma: no cover - network/setup failures
        raise RuntimeError(f"failed to pull Ollama model '{model}': {exc}") from exc

    stream = result if isinstance(result, Iterator) else (result,)
    for chunk in stream:
        status = getattr(chunk, "status", None)
        if status is None and isinstance(chunk, dict):
            status = chunk.get("status")
        completed = getattr(chunk, "completed", None)
        total = getattr(chunk, "total", None)
        if isinstance(chunk, dict):
            completed = chunk.get("completed", completed)
            total = chunk.get("total", total)
        if status:
            if isinstance(completed, (int, float)) and isinstance(total, (int, float)) and total:
                pct = int((completed / total) * 100)
                print(f"[ollama] {status} ({pct}%)", file=sys.stderr, flush=True)
            else:
                print(f"[ollama] {status}", file=sys.stderr, flush=True)
    print(f"[ollama] Model '{model}' is ready.", file=sys.stderr)


def _prompt_for_installed_model(installed: Sequence[str]) -> str | None:
    if not installed:
        print("No local Ollama models found.")
        return None
    print("Installed Ollama models:")
    for idx, name in enumerate(installed, start=1):
        print(f"  {idx}. {name}")
    while True:
        choice = input("Select a model by number or name (Enter to cancel): ").strip()
        if not choice:
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(installed):
                return installed[idx - 1]
        else:
            for name in installed:
                if name.lower() == choice.lower():
                    return name
        print("Invalid selection. Please try again.")


def _resolve_missing_ollama_model(
    adapter: OllamaProviderAdapter,
    *,
    missing_model: str,
    installed: Sequence[str],
    debug: bool,
) -> str:
    if not sys.stdin.isatty():
        raise RuntimeError(
            f"Ollama model '{missing_model}' is not installed. "
            f"Run 'ollama pull {missing_model}' or set AI_MODEL to a downloaded model."
        )

    print(f"Ollama model '{missing_model}' is not installed locally.")
    while True:
        choice = input("Pull it now? [P]ull / [S]elect installed model / [Q]uit: ").strip().lower()
        if choice in {"", "p", "pull", "y", "yes"}:
            _pull_ollama_model(adapter, missing_model)
            installed = adapter.list_models(debug=debug)
            if missing_model in installed:
                return missing_model
            print(
                f"Model '{missing_model}' still missing after pull. "
                "You can retry pulling or select another installed model."
            )
        elif choice in {"s", "select"}:
            selected = _prompt_for_installed_model(installed)
            if selected:
                print(f"Using installed model '{selected}'.")
                return selected
            print("No model selected; please choose an option.")
            installed = adapter.list_models(debug=debug)
        elif choice in {"q", "quit", "n", "no"}:
            raise RuntimeError("aborted: no Ollama model selected.")
        else:
            print("Please respond with 'p', 's', or 'q'.")


def ensure_ollama_model(model: str, *, debug: bool, log_prefix: str = "[ollama]") -> str:
    """Ensure an Ollama model is available, prompting to pull or select if missing.
    
    Returns the validated/selected model name.
    Raises RuntimeError if model validation fails or user aborts.
    
    Notes:
    - qwen2.5-coder:32b: Recommended for coding tasks with tool calling; may return JSON-formatted responses
      which are automatically unwrapped by the adapter.
    - gpt-oss: Older baseline model; works but less powerful than qwen2.5-coder.
    """
    prepare_runtime(debug=debug, log_prefix=log_prefix)
    adapter = OllamaProviderAdapter(client=ollama.Client())
    try:
        installed = adapter.list_models(debug=debug)
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError(f"failed to list Ollama models: {exc}") from exc
    if model in installed:
        return model
    return _resolve_missing_ollama_model(adapter, missing_model=model, installed=installed, debug=debug)

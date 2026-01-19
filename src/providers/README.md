# Providers

This folder contains provider-specific code in a predictable layout.

## Current structure

- `base.py` – the `ProviderAdapter` protocol used by the core agent runner
- `_raw_tools.py` – provider-agnostic tool specs (providers map these into their schema)
- `tool_schemas.py` – shared helpers to map raw tool specs into provider-specific schema shapes
- `openai/` – OpenAI implementation
- `ollama/` – Ollama implementation
- `copilot/` – GitHub Models (Copilot) implementation

## Adding a new provider

1. Create a new package, e.g. `providers/foo/`.
2. Implement an adapter in `providers/foo/adapter.py` that satisfies `ProviderAdapter`:
   - `call_model(model, tools, context, debug) -> response`
   - `parse_response(response, debug) -> ParseResult`
3. (Optional) Add runtime helpers in `providers/foo/runtime.py` if the provider needs setup.
4. Implement a tool schema mapper in `providers/foo/tools.py` that exports `build_tools(prefix)`.
   - Prefer using helpers in `providers/tool_schemas.py` when the provider uses a common tool schema.
5. Wire it into the REPL by adding a `ProviderEntry` in `src/entrypoints/ai.py`.

That’s it — the core agent loop should not need changes.
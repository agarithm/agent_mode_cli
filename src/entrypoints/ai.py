from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import ollama  # noqa: F401

from core.agent_runner import AgentRunnerConfig, ProviderEntry, run_agent_repl
from core.cli_help import handle_common_flags
from core.system_prompt import build_internal_system_prompt
from core.ui_runners import run_fullscreen, run_inline
from providers.ollama.adapter import OllamaProviderAdapter
from providers.ollama.runtime import prepare_runtime
from providers.ollama.tools import build_tools as build_ollama_tools
from providers.ollama.validation import ensure_ollama_model
from providers.github.adapter import GitHubProviderAdapter
from providers.github.runtime import create_github_models_client
from providers.github.tools import build_tools as build_github_tools
from providers.openai.adapter import OpenAIProviderAdapter
from providers.openai.runtime import create_openai_client
from providers.openai.tools import build_tools as build_openai_tools


INTERNAL_SYSTEM_PROMPT = build_internal_system_prompt("AI")
CONTAINER_ENV_FLAG = "AI_IN_CONTAINER"
DEFAULT_CONTAINER_IMAGE = os.getenv("AI_CONTAINER_IMAGE", "localhost/agent-mode-dev:latest")
DEFAULT_PODMAN_BIN = os.getenv("PODMAN_BIN", "docker")


def _is_truthy(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


def _running_inside_container() -> bool:
    return _is_truthy(os.getenv(CONTAINER_ENV_FLAG, ""))


def _container_launch_disabled() -> bool:
    return _is_truthy(os.getenv("AI_CONTAINER_DISABLE", ""))


_ENV_PASSTHROUGH_DENYLIST = {CONTAINER_ENV_FLAG, "HOME"}


def _collect_env_passthrough() -> list[str]:
    flags: list[str] = []
    for key, value in os.environ.items():
        if key in _ENV_PASSTHROUGH_DENYLIST:
            continue
        if "=" in key or "\n" in key or "\x00" in key:
            continue
        if isinstance(value, str) and ("\n" in value or "\x00" in value):
            continue
        flags.extend(["--env", f"{key}={value}"])
    flags.extend(["--env", f"{CONTAINER_ENV_FLAG}=1"])
    return flags


def _maybe_run_inside_container(argv: list[str]) -> None:
    if _running_inside_container() or _container_launch_disabled():
        return

    podman_bin = os.getenv("PODMAN_BIN", DEFAULT_PODMAN_BIN)
    if shutil.which(podman_bin) is None:
        print(
            f"warning: '{podman_bin}' not found; running natively without container",
            file=sys.stderr,
        )
        return

    cwd = os.getcwd()
    if not cwd:
        return

    container_image = os.getenv("AI_CONTAINER_IMAGE", DEFAULT_CONTAINER_IMAGE)
    stdio_flags = ["--interactive"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        stdio_flags.append("--tty")

    # Docker-specific vs Podman-specific flags
    is_podman = "podman" in podman_bin.lower()
    userns_flags = ["--userns=keep-id"] if is_podman else ["--user", f"{os.getuid()}:{os.getgid()}"]
    volume_suffix = ":rw,Z" if is_podman else ":rw"
    gpu_flags = ["--gpus", "all"] if not is_podman else ["--device", "nvidia.com/gpu=all"]

    cmd = [
        podman_bin,
        "run",
        "--rm",
        *userns_flags,
        *gpu_flags,
        *stdio_flags,
        *_collect_env_passthrough(),
        "--volume",
        f"{cwd}:{cwd}{volume_suffix}",
        "--workdir",
        cwd,
        container_image,
        "ai",
        *argv,
    ]

    print(
        f"[ai] launching container '{container_image}' via {podman_bin} for current directory",
        file=sys.stderr,
    )
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError:
        print(
            f"error: unable to execute '{podman_bin}' for container launch",
            file=sys.stderr,
        )
        raise

    raise SystemExit(result.returncode)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    _maybe_run_inside_container(argv)
    from version import __version__

    def _extract_inline_flag(args: list[str]) -> tuple[bool, list[str]]:
        inline = False
        remaining: list[str] = []
        i = 0
        while i < len(args):
            token = args[i]
            if token == "--inline":
                inline = True
                i += 1
                continue
            remaining.append(token)
            i += 1
        return inline, remaining

    inline, argv = _extract_inline_flag(argv)

    debug = os.getenv("AI_DEBUG", "").lower() in ("1", "true", "yes", "on")
    env_provider = os.getenv("AI_PROVIDER", "ollama").strip().lower() or "ollama"
    base_ollama_model = os.getenv("AI_MODEL", "gpt-oss:latest")

    flag_exit = handle_common_flags(
        argv,
        usage="ai [--inline] [provider] [model] [prompt...]",
        description=(
            "Agent-mode CLI (full REPL) with dynamic provider switching. "
            "Default provider is Ollama (local)."
        ),
        env_lines=(
            "AI_PROVIDER     optional (default: ollama)",
            "AI_MODEL        optional (default: gpt-oss:latest; applies to current provider)",
            "AI_DEBUG        optional (1/true enables debug)",
            "AI_PROMPT_FILE  optional (default: ~/.ai_prompt)",
            "OPENAI_API_KEY  required for OpenAI provider",
            "GITHUB_TOKEN    required for GitHub Models provider",
            "OLLAMA_HOST     optional (use remote Ollama)",
            "First CLI args  optional magic keywords for provider/model",
        ),
        version=__version__,
    )
    if flag_exit is not None:
        return flag_exit

    def _create_ollama_adapter() -> OllamaProviderAdapter:
        try:
            import ollama as _ollama  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ollama Python package is not available. Run inside the dev container or install it manually."
            ) from exc
        client = _ollama.Client()
        return OllamaProviderAdapter(client=client)

    def _create_openai_adapter() -> OpenAIProviderAdapter:
        client = create_openai_client()
        return OpenAIProviderAdapter(client=client)

    def _create_github_adapter() -> GitHubProviderAdapter:
        client = create_github_models_client()
        return GitHubProviderAdapter(client=client)

    def _build_providers(default_ollama_model: str, overrides: dict[str, str] | None = None) -> dict[str, ProviderEntry]:
        overrides = overrides or {}
        return {
            "ollama": ProviderEntry(
                name="ollama",
                description="Local Ollama (default)",
                default_model=overrides.get("ollama") or default_ollama_model,
                build_tools=build_ollama_tools,
                create_adapter=_create_ollama_adapter,
                prepare_runtime=lambda debug: prepare_runtime(debug=debug, log_prefix="[ai]"),
                validate_model=lambda model, debug: ensure_ollama_model(model, debug=debug, log_prefix="[ai]"),
            ),
            "github": ProviderEntry(
                name="github",
                description="GitHub Models (requires GITHUB_TOKEN)",
                default_model=overrides.get("github") or os.getenv("AI_GITHUB_MODEL", "xai/grok-3"),
                build_tools=build_github_tools,
                create_adapter=_create_github_adapter,
                prepare_runtime=None,
                fallback_providers=(
                    os.getenv("AI_GITHUB_FALLBACK_PRIMARY", "openai"),
                    os.getenv("AI_GITHUB_FALLBACK_SECONDARY", "ollama"),
                ),
            ),
            "openai": ProviderEntry(
                name="openai",
                description="OpenAI (requires OPENAI_API_KEY)",
                default_model=overrides.get("openai") or os.getenv("AI_OPENAI_MODEL", "gpt-5.2"),
                build_tools=build_openai_tools,
                create_adapter=_create_openai_adapter,
                prepare_runtime=None,
                fallback_providers=(
                    os.getenv("AI_OPENAI_FALLBACK_PRIMARY", "github"),
                    os.getenv("AI_OPENAI_FALLBACK_SECONDARY", "ollama"),
                ),
            ),
        }

    cli_prevalidated_models: dict[str, str] = {}
    cli_adapter_cache: dict[str, object] = {}

    def _extract_cli_overrides(raw_args: list[str], providers_map: dict[str, ProviderEntry]) -> tuple[Optional[str], Optional[str], list[str]]:
        remaining = list(raw_args)
        if not remaining:
            return None, None, []

        candidate_provider = (remaining[0] or "").strip().lower()
        if candidate_provider not in providers_map:
            return None, None, remaining

        provider_override = candidate_provider
        tokens_after_provider = remaining[1:]

        if not tokens_after_provider:
            return provider_override, None, []

        candidate_model = tokens_after_provider[0]

        def _candidate_is_valid_model(provider_name: str, candidate: str) -> tuple[bool, Optional[str]]:
            entry = providers_map[provider_name]
            normalized = (candidate or "").strip()
            if not normalized:
                return False, None
            if entry.validate_model is not None:
                try:
                    validated = entry.validate_model(normalized, debug)
                    cli_prevalidated_models[provider_name] = validated
                    return True, validated
                except Exception as exc:
                    if debug:
                        print(
                            f"[debug] CLI model '{normalized}' rejected for provider '{provider_name}': {exc}",
                            file=sys.stderr,
                        )
                    return False, None

            adapter = cli_adapter_cache.get(provider_name)
            if adapter is None:
                try:
                    if entry.prepare_runtime is not None:
                        entry.prepare_runtime(debug)
                    adapter = entry.create_adapter()
                    cli_adapter_cache[provider_name] = adapter
                except Exception as exc:
                    if debug:
                        print(
                            f"[debug] CLI could not initialize adapter for provider '{provider_name}': {exc}",
                            file=sys.stderr,
                        )
                    return False, None
            try:
                available = [m.strip() for m in adapter.list_models(debug=debug)]  # type: ignore[attr-defined]
            except Exception as exc:
                if debug:
                    print(
                        f"[debug] CLI could not list models for provider '{provider_name}': {exc}",
                        file=sys.stderr,
                    )
                return False, None
            available = [m for m in available if m]
            if normalized in available:
                return True, normalized
            if debug:
                print(
                    f"[debug] CLI candidate '{normalized}' not found in provider '{provider_name}' catalog",
                    file=sys.stderr,
                )
            return False, None

        is_valid_model, normalized_model = _candidate_is_valid_model(provider_override, candidate_model)
        if is_valid_model and normalized_model:
            return provider_override, normalized_model, tokens_after_provider[1:]
        return provider_override, None, tokens_after_provider

    providers = _build_providers(base_ollama_model)
    provider_override, model_override, prompt_tokens = _extract_cli_overrides(argv, providers)

    provider_model_overrides: dict[str, str] = {}
    if provider_override and model_override:
        provider_model_overrides[provider_override] = model_override

    if provider_override:
        os.environ["AI_PROVIDER"] = provider_override
    if provider_override and model_override:
        os.environ["AI_MODEL"] = model_override

    initial_provider = provider_override or env_provider
    default_ollama_model = cli_prevalidated_models.get("ollama") or base_ollama_model

    if initial_provider == "ollama" and "ollama" not in cli_prevalidated_models:
        try:
            ensured_model = ensure_ollama_model(
                provider_model_overrides.get("ollama") or default_ollama_model,
                debug=debug,
                log_prefix="[ai]",
            )
            default_ollama_model = ensured_model
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    providers = _build_providers(default_ollama_model, overrides=provider_model_overrides)
    initial_line = " ".join(prompt_tokens) if prompt_tokens else None

    runner_config = AgentRunnerConfig(
        agent_name="AI",
        env_prefix="AI",
        debug_env="AI_DEBUG",
        model_env="AI_MODEL",
        initial_debug=debug,
        initial_model=default_ollama_model,
        prompt_file_env="AI_PROMPT_FILE",
        prompt_file_default=".ai_prompt",
        internal_system_prompt=INTERNAL_SYSTEM_PROMPT,
        catch_runtime_errors=True,
    )

    initial_model_banner = (os.getenv("AI_MODEL") or "").strip()
    if not initial_model_banner:
        entry = providers.get(initial_provider)
        initial_model_banner = (entry.default_model if entry else "") or "unknown"
    print(
        f"[ai] Provider: {initial_provider} {initial_model_banner}",
    )

    try:
        repl_runner = run_inline if inline else run_fullscreen
        return run_agent_repl(
            providers=providers,
            initial_provider=initial_provider,
            config=runner_config,
            initial_line=initial_line,
            repl_runner=repl_runner,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

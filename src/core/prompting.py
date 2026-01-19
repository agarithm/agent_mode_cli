from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence


class PromptBackend(Protocol):
    def confirm(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        approve_all: bool,
        debug: bool = False,
    ) -> str:
        ...

    def select_one(
        self,
        *,
        title: str,
        options: Sequence[str],
        default: Optional[str] = None,
        allow_cancel: bool = True,
    ) -> Optional[str]:
        ...

    def prompt_text(
        self,
        *,
        title: str,
        placeholder: str = "",
        default: str = "",
        allow_cancel: bool = True,
    ) -> Optional[str]:
        ...


_BACKEND: Optional[PromptBackend] = None


def set_prompt_backend(backend: PromptBackend) -> None:
    global _BACKEND
    _BACKEND = backend


def clear_prompt_backend() -> None:
    global _BACKEND
    _BACKEND = None


def get_prompt_backend() -> Optional[PromptBackend]:
    return _BACKEND


def has_prompt_backend() -> bool:
    return _BACKEND is not None


def confirm(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    approve_all: bool,
    debug: bool = False,
) -> Optional[str]:
    backend = _BACKEND
    if backend is None:
        return None
    return backend.confirm(tool_name=tool_name, arguments=arguments, approve_all=approve_all, debug=debug)


def select_one(
    *,
    title: str,
    options: Sequence[str],
    default: Optional[str] = None,
    allow_cancel: bool = True,
) -> Optional[str]:
    backend = _BACKEND
    if backend is None:
        return None
    return backend.select_one(title=title, options=options, default=default, allow_cancel=allow_cancel)


def prompt_text(
    *,
    title: str,
    placeholder: str = "",
    default: str = "",
    allow_cancel: bool = True,
) -> Optional[str]:
    backend = _BACKEND
    if backend is None:
        return None
    return backend.prompt_text(title=title, placeholder=placeholder, default=default, allow_cancel=allow_cancel)

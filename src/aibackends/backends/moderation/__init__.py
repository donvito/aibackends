from __future__ import annotations

from aibackends.backends.moderation._base import BaseModerationBackend
from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import normalize_name

_MODERATION_BACKENDS: dict[str, BaseModerationBackend] = {}
_BUILTINS_REGISTERED = False


def register_moderation_backend(backend: BaseModerationBackend) -> None:
    for name in backend.names:
        _MODERATION_BACKENDS[normalize_name(name)] = backend


def get_moderation_backend(name: str) -> BaseModerationBackend:
    _ensure_builtin_moderation_backends_registered()
    try:
        return _MODERATION_BACKENDS[normalize_name(name)]
    except KeyError as exc:
        raise TaskExecutionError(f"Unsupported moderation backend: {name}") from exc


def list_moderation_backends() -> list[str]:
    _ensure_builtin_moderation_backends_registered()
    return sorted({backend.name for backend in _MODERATION_BACKENDS.values()})


def _ensure_builtin_moderation_backends_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from aibackends.backends.moderation.gliguard import GLIGUARD_BACKEND

    register_moderation_backend(GLIGUARD_BACKEND)
    _BUILTINS_REGISTERED = True


__all__ = [
    "BaseModerationBackend",
    "get_moderation_backend",
    "list_moderation_backends",
    "register_moderation_backend",
]

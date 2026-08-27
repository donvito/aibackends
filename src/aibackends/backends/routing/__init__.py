from __future__ import annotations

from aibackends.backends.routing._base import BaseRoutingBackend
from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import normalize_name

_ROUTING_BACKENDS: dict[str, BaseRoutingBackend] = {}
_BUILTINS_REGISTERED = False


def register_routing_backend(backend: BaseRoutingBackend) -> None:
    for name in backend.names:
        _ROUTING_BACKENDS[normalize_name(name)] = backend


def get_routing_backend(name: str) -> BaseRoutingBackend:
    _ensure_builtin_routing_backends_registered()
    try:
        return _ROUTING_BACKENDS[normalize_name(name)]
    except KeyError as exc:
        raise TaskExecutionError(f"Unsupported routing backend: {name}") from exc


def list_routing_backends() -> list[str]:
    _ensure_builtin_routing_backends_registered()
    return sorted({backend.name for backend in _ROUTING_BACKENDS.values()})


def _ensure_builtin_routing_backends_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from aibackends.backends.routing.lfm2_router import LFM2_PROMPT_ROUTER_BACKEND

    register_routing_backend(LFM2_PROMPT_ROUTER_BACKEND)
    _BUILTINS_REGISTERED = True


__all__ = [
    "BaseRoutingBackend",
    "get_routing_backend",
    "list_routing_backends",
    "register_routing_backend",
]

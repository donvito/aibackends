from __future__ import annotations

from aibackends.backends.extraction._base import BaseExtractionBackend
from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import normalize_name

_EXTRACTION_BACKENDS: dict[str, BaseExtractionBackend] = {}
_BUILTINS_REGISTERED = False


def register_extraction_backend(backend: BaseExtractionBackend) -> None:
    for name in backend.names:
        _EXTRACTION_BACKENDS[normalize_name(name)] = backend


def get_extraction_backend(name: str) -> BaseExtractionBackend:
    _ensure_builtin_extraction_backends_registered()
    try:
        return _EXTRACTION_BACKENDS[normalize_name(name)]
    except KeyError as exc:
        raise TaskExecutionError(f"Unsupported extraction backend: {name}") from exc


def list_extraction_backends() -> list[str]:
    _ensure_builtin_extraction_backends_registered()
    return sorted({backend.name for backend in _EXTRACTION_BACKENDS.values()})


def _ensure_builtin_extraction_backends_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from aibackends.backends.extraction.gliner25 import GLINER25_BACKEND

    register_extraction_backend(GLINER25_BACKEND)
    _BUILTINS_REGISTERED = True


__all__ = [
    "BaseExtractionBackend",
    "get_extraction_backend",
    "list_extraction_backends",
    "register_extraction_backend",
]

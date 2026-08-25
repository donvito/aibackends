from __future__ import annotations

from aibackends.backends.information_extraction._base import (
    BaseInformationExtractionBackend,
    EntityTypes,
)
from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import normalize_name

_INFORMATION_EXTRACTION_BACKENDS: dict[str, BaseInformationExtractionBackend] = {}
_BUILTINS_REGISTERED = False


def register_information_extraction_backend(
    backend: BaseInformationExtractionBackend,
) -> None:
    for name in backend.names:
        _INFORMATION_EXTRACTION_BACKENDS[normalize_name(name)] = backend


def get_information_extraction_backend(name: str) -> BaseInformationExtractionBackend:
    _ensure_builtin_information_extraction_backends_registered()
    try:
        return _INFORMATION_EXTRACTION_BACKENDS[normalize_name(name)]
    except KeyError as exc:
        raise TaskExecutionError(
            f"Unsupported information extraction backend: {name}"
        ) from exc


def list_information_extraction_backends() -> list[str]:
    _ensure_builtin_information_extraction_backends_registered()
    return sorted(
        {backend.name for backend in _INFORMATION_EXTRACTION_BACKENDS.values()}
    )


def _ensure_builtin_information_extraction_backends_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    from aibackends.backends.information_extraction.gliner25 import GLINER25_BACKEND

    register_information_extraction_backend(GLINER25_BACKEND)
    _BUILTINS_REGISTERED = True


__all__ = [
    "BaseInformationExtractionBackend",
    "EntityTypes",
    "get_information_extraction_backend",
    "list_information_extraction_backends",
    "register_information_extraction_backend",
]

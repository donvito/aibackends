from __future__ import annotations

from aibackends.core.exceptions import ConfigurationError
from aibackends.core.registry import ModelSupportSpec, discover_specs, normalize_name

_MODEL_SUPPORT: dict[str, ModelSupportSpec] = {}
_BUILTINS_REGISTERED = False


def register_model_support(spec: ModelSupportSpec) -> None:
    _MODEL_SUPPORT[normalize_name(spec.runtime)] = spec


def get_model_support(runtime: str | None) -> ModelSupportSpec | None:
    if runtime is None:
        return None
    _ensure_builtin_model_support_registered()
    return _MODEL_SUPPORT.get(normalize_name(runtime))


def _ensure_builtin_model_support_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for spec in discover_specs("aibackends.model_support", "MODEL_SUPPORT_SPEC"):
        if not isinstance(spec, ModelSupportSpec):
            raise ConfigurationError(f"Invalid model support spec: {spec!r}")
        register_model_support(spec)
    _BUILTINS_REGISTERED = True


__all__ = [
    "get_model_support",
    "register_model_support",
]

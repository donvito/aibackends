from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from aibackends.core.exceptions import ConfigurationError, RuntimeNotConfiguredError
from aibackends.core.registry import RuntimeSpec, discover_specs, normalize_name
from aibackends.core.types import RuntimeConfig

if False:  # pragma: no cover
    from aibackends.core.runtimes.base import BaseRuntime

RuntimeFactory = Callable[[RuntimeConfig], "BaseRuntime"]

_GLOBAL_CONFIG = RuntimeConfig()
_RUNTIME_FACTORIES: dict[str, RuntimeFactory] = {}
_BUILTINS_REGISTERED = False


def register_runtime(name: str, factory: RuntimeFactory) -> None:
    _RUNTIME_FACTORIES[normalize_name(name)] = factory


def _ensure_builtin_runtimes_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    for spec in discover_specs("aibackends.core.runtimes", "RUNTIME_SPEC"):
        if not isinstance(spec, RuntimeSpec):
            raise ConfigurationError(f"Invalid runtime spec: {spec!r}")
        for name in spec.names:
            register_runtime(name, spec.factory)
    _BUILTINS_REGISTERED = True


def configure(**kwargs: Any) -> RuntimeConfig:
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = resolve_runtime_config(kwargs)
    return get_settings()


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise ConfigurationError(f"Config file does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("Config file must contain a top-level mapping.")
    return configure(**raw)


def get_settings() -> RuntimeConfig:
    return _GLOBAL_CONFIG.model_copy(deep=True)


def reset_config() -> None:
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = RuntimeConfig()


def resolve_runtime_config(
    overrides: dict[str, Any] | RuntimeConfig | None = None,
) -> RuntimeConfig:
    base_data = get_settings().model_dump()
    incoming = (
        overrides.model_dump(exclude_none=True)
        if isinstance(overrides, RuntimeConfig)
        else overrides or {}
    )
    incoming = {key: value for key, value in incoming.items() if value is not None}
    incoming_extra = incoming.pop("extra_options", {})
    merged = {**base_data, **incoming}
    merged["extra_options"] = {**base_data.get("extra_options", {}), **incoming_extra}
    return RuntimeConfig.model_validate(merged)


def get_runtime(overrides: dict[str, Any] | RuntimeConfig | None = None) -> BaseRuntime:
    _ensure_builtin_runtimes_registered()
    config = resolve_runtime_config(overrides)
    if not config.runtime:
        raise RuntimeNotConfiguredError(
            "No runtime configured. Call `aibackends.configure(runtime=..., model=...)` first."
        )
    try:
        factory = _RUNTIME_FACTORIES[normalize_name(config.runtime)]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown runtime: {config.runtime}") from exc
    return factory(config)

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from aibackends.core.exceptions import ConfigurationError, RuntimeNotConfiguredError
from aibackends.core.registry import ModelRef, RuntimeSpec, discover_specs, normalize_name
from aibackends.core.types import RuntimeConfig

if False:  # pragma: no cover
    from aibackends.core.runtimes.base import BaseRuntime

RuntimeFactory = Callable[[RuntimeConfig], "BaseRuntime"]

_GLOBAL_CONFIG = RuntimeConfig()
_RUNTIME_FACTORIES: dict[str, RuntimeFactory] = {}
_RUNTIME_SPECS: dict[str, RuntimeSpec] = {}
_RUNTIME_CACHE: dict[tuple[Any, ...], BaseRuntime] = {}
_PER_CALL_CONFIG_FIELDS = (
    "temperature",
    "max_tokens",
    "timeout",
    "max_retries",
    "on_step_complete",
)
_BUILTINS_REGISTERED = False


def register_runtime(name: str, factory: RuntimeFactory) -> RuntimeSpec:
    spec = RuntimeSpec(name=name, factory=factory)
    register_runtime_spec(spec)
    return spec


def register_runtime_spec(spec: RuntimeSpec) -> None:
    for name in spec.names:
        _RUNTIME_FACTORIES[normalize_name(name)] = spec.factory
        _RUNTIME_SPECS[normalize_name(name)] = spec


def _ensure_builtin_runtimes_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    for spec in discover_specs("aibackends.core.runtimes", "RUNTIME_SPEC"):
        if not isinstance(spec, RuntimeSpec):
            raise ConfigurationError(f"Invalid runtime spec: {spec!r}")
        register_runtime_spec(spec)
    _BUILTINS_REGISTERED = True


def get_runtime_spec(name: str) -> RuntimeSpec:
    _ensure_builtin_runtimes_registered()
    try:
        return _RUNTIME_SPECS[normalize_name(name)]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown runtime: {name}") from exc


def available_runtimes() -> dict[str, RuntimeSpec]:
    _ensure_builtin_runtimes_registered()
    names = sorted({spec.name for spec in _RUNTIME_SPECS.values()})
    return {name: get_runtime_spec(name) for name in names}


def ensure_runtime_spec(value: Any, *, param_name: str = "runtime") -> RuntimeSpec | None:
    if value is None or isinstance(value, RuntimeSpec):
        return value
    raise TypeError(
        f"{param_name} must be a RuntimeSpec. "
        "Use `aibackends.runtimes.*` or `get_runtime_spec(name)`."
    )


def ensure_model_ref(value: Any, *, param_name: str = "model") -> ModelRef | None:
    if value is None or isinstance(value, ModelRef):
        return value
    raise TypeError(
        f"{param_name} must be a ModelRef. "
        "Use `aibackends.models.*` or `ModelRef(name=...)`."
    )


def parse_runtime_text(value: str | None) -> RuntimeSpec | None:
    if value is None:
        return None
    return get_runtime_spec(value)


def parse_model_text(value: str | None) -> ModelRef | None:
    if value is None:
        return None
    return ModelRef(name=value)


def resolve_python_runtime_config(
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    overrides: dict[str, Any] | None = None,
) -> RuntimeConfig:
    runtime = ensure_runtime_spec(runtime)
    model = ensure_model_ref(model)
    incoming = dict(overrides or {})
    if "runtime" in incoming:
        runtime = ensure_runtime_spec(incoming.pop("runtime"))
    if "model" in incoming:
        model = ensure_model_ref(incoming.pop("model"))
    return resolve_runtime_config(
        {
            "runtime": runtime,
            "model": model,
            **incoming,
        }
    )


def configure(
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **kwargs: Any,
) -> RuntimeConfig:
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = resolve_python_runtime_config(
        runtime=runtime,
        model=model,
        overrides=kwargs,
    )
    return get_settings()


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path).expanduser()
    if not config_path.exists():
        raise ConfigurationError(f"Config file does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError("Config file must contain a top-level mapping.")
    parsed = dict(raw)
    if "runtime" in parsed:
        parsed["runtime"] = parse_runtime_text(parsed["runtime"])
    if "model" in parsed:
        parsed["model"] = parse_model_text(parsed["model"])
    return configure(**parsed)


def get_settings() -> RuntimeConfig:
    return _GLOBAL_CONFIG.model_copy(deep=True)


def reset_config() -> None:
    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = RuntimeConfig()
    clear_runtime_cache()


def clear_runtime_cache() -> None:
    """Drop all cached runtime instances, releasing any loaded models."""
    _RUNTIME_CACHE.clear()


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


def _runtime_cache_key(config: RuntimeConfig) -> tuple[Any, ...]:
    # Only fields that affect what gets loaded into memory. Per-call generation
    # parameters (temperature, max_tokens, ...) are deliberately excluded so a
    # tweak does not force a full model reload.
    return (
        normalize_name(config.runtime or ""),
        config.model,
        config.model_path,
        config.adapter,
        config.device,
        config.load_in_4bit,
        config.cache_dir,
        config.prompt_format,
        config.chat_template,
        config.chat_template_path,
        config.base_url,
        config.api_key,
        json.dumps(config.extra_options, sort_keys=True, default=str),
    )


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
    if not config.reuse_runtime:
        return factory(config)
    key = _runtime_cache_key(config)
    cached = _RUNTIME_CACHE.get(key)
    if cached is not None:
        # Refresh per-call parameters while keeping the already-loaded model
        # client. Only fields excluded from the cache key are copied over, so
        # runtime-applied config transforms (e.g. transformers model profiles)
        # are preserved.
        cached.config = cached.config.model_copy(
            update={field: getattr(config, field) for field in _PER_CALL_CONFIG_FIELDS}
        )
        return cached
    runtime = factory(config)
    _RUNTIME_CACHE[key] = runtime
    return runtime


def preload(overrides: dict[str, Any] | RuntimeConfig | None = None) -> BaseRuntime:
    """Resolve the runtime and load its model ahead of the first request."""
    runtime = get_runtime(overrides)
    runtime.preload()
    return runtime

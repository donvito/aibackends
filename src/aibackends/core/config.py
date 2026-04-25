from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from aibackends.core.exceptions import ConfigurationError, RuntimeNotConfiguredError
from aibackends.core.types import RuntimeConfig

if False:  # pragma: no cover
    from aibackends.core.runtimes.base import BaseRuntime

RuntimeFactory = Callable[[RuntimeConfig], "BaseRuntime"]

_GLOBAL_CONFIG = RuntimeConfig()
_RUNTIME_FACTORIES: dict[str, RuntimeFactory] = {}
_BUILTINS_REGISTERED = False


def register_runtime(name: str, factory: RuntimeFactory) -> None:
    _RUNTIME_FACTORIES[name] = factory


def _ensure_builtin_runtimes_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from aibackends.core.runtimes.anthropic import AnthropicRuntime
    from aibackends.core.runtimes.groq import GroqRuntime
    from aibackends.core.runtimes.llamacpp import LlamaCppRuntime
    from aibackends.core.runtimes.lmstudio import LMStudioRuntime
    from aibackends.core.runtimes.ollama import OllamaRuntime
    from aibackends.core.runtimes.together import TogetherRuntime
    from aibackends.core.runtimes.transformers import TransformersRuntime

    register_runtime("anthropic", AnthropicRuntime)
    register_runtime("groq", GroqRuntime)
    register_runtime("llamacpp", LlamaCppRuntime)
    register_runtime("lmstudio", LMStudioRuntime)
    register_runtime("ollama", OllamaRuntime)
    register_runtime("together", TogetherRuntime)
    register_runtime("transformers", TransformersRuntime)
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
        factory = _RUNTIME_FACTORIES[config.runtime]
    except KeyError as exc:
        raise ConfigurationError(f"Unknown runtime: {config.runtime}") from exc
    return factory(config)

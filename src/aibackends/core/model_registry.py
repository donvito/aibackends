from __future__ import annotations

from aibackends.core.registry import TransformerModelProfile, discover_specs, normalize_name
from aibackends.core.types import RuntimeConfig

MODEL_REGISTRY: dict[str, str] = {}
RUNTIME_MODEL_REGISTRY: dict[str, dict[str, str]] = {}

_MODEL_PROFILES: dict[tuple[str | None, str], TransformerModelProfile] = {}
_PROFILES_REGISTERED = False


def register_model_profile(profile: TransformerModelProfile) -> None:
    runtime = normalize_name(profile.runtime) if profile.runtime else None
    for name in profile.names:
        normalized = normalize_name(name)
        _MODEL_PROFILES[(runtime, normalized)] = profile
        if runtime is None:
            MODEL_REGISTRY[normalized] = profile.model_id
        else:
            RUNTIME_MODEL_REGISTRY.setdefault(runtime, {})[normalized] = profile.model_id


def _ensure_model_profiles_registered() -> None:
    global _PROFILES_REGISTERED
    if _PROFILES_REGISTERED:
        return
    for profile in discover_specs("aibackends.models", "MODEL_PROFILE"):
        register_model_profile(profile)
    for profile in discover_specs("aibackends.models", "MODEL_PROFILES"):
        register_model_profile(profile)
    _PROFILES_REGISTERED = True


def resolve_model_alias(name: str | None, *, runtime: str | None = None) -> str | None:
    profile = resolve_model_profile(name, runtime=runtime)
    if profile is not None:
        return profile.model_id
    if not name:
        return None
    return name


def resolve_model_profile(
    name: str | None,
    *,
    runtime: str | None = None,
) -> TransformerModelProfile | None:
    if not name:
        return None
    _ensure_model_profiles_registered()
    normalized_name = normalize_name(name)
    normalized_runtime = normalize_name(runtime) if runtime else None
    if normalized_runtime is not None:
        runtime_profile = _MODEL_PROFILES.get((normalized_runtime, normalized_name))
        if runtime_profile is not None:
            return runtime_profile
    shared_profile = _MODEL_PROFILES.get((None, normalized_name))
    if shared_profile is not None:
        return shared_profile
    return _MODEL_PROFILES.get(("transformers", normalized_name))


def apply_transformer_model_profile(config: RuntimeConfig) -> RuntimeConfig:
    profile = resolve_model_profile(config.model, runtime="transformers")
    if profile is None:
        return config

    updates: dict[str, object] = {}
    updates["model"] = profile.model_id
    if config.chat_template is None and config.chat_template_path is None:
        if profile.chat_template is not None:
            updates["chat_template"] = profile.chat_template
        elif profile.chat_template_path is not None:
            updates["chat_template_path"] = profile.chat_template_path
    if profile.prompt_format is not None and config.prompt_format == "auto":
        updates["prompt_format"] = profile.prompt_format
    if profile.generation_defaults:
        updates["extra_options"] = {
            **profile.generation_defaults,
            **config.extra_options,
        }
    if not updates:
        return config
    return config.model_copy(update=updates)

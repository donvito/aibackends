from __future__ import annotations

from aibackends.core.exceptions import ModelResolutionError
from aibackends.core.registry import (
    ModelRef,
    ModelRefLike,
    RuntimeRefLike,
    RuntimeSpec,
    TransformerModelProfile,
    discover_specs,
    normalize_name,
)
from aibackends.core.types import RuntimeConfig

MODEL_REGISTRY: dict[str, str] = {}
RUNTIME_MODEL_REGISTRY: dict[str, dict[str, str]] = {}

_MODEL_PROFILES: dict[tuple[str | None, str], TransformerModelProfile] = {}
_MODEL_REFS: dict[str, ModelRef] = {}
_MODEL_REF_ALIASES: dict[str, ModelRef] = {}
_PROFILES_REGISTERED = False
_SHARED_MODEL_RUNTIMES = {"llamacpp", "transformers"}


def register_model_profile(profile: TransformerModelProfile) -> None:
    runtime = normalize_name(profile.runtime) if profile.runtime else None
    model_ref = _MODEL_REFS.setdefault(
        normalize_name(profile.name),
        ModelRef(name=profile.name),
    )
    for name in profile.names:
        normalized = normalize_name(name)
        _MODEL_PROFILES[(runtime, normalized)] = profile
        _MODEL_REF_ALIASES[normalized] = model_ref
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


def get_model_ref(name: str) -> ModelRef:
    _ensure_model_profiles_registered()
    try:
        return _MODEL_REF_ALIASES[normalize_name(name)]
    except KeyError as exc:
        raise ModelResolutionError(f"Unknown supported model: {name}") from exc


def available_models(runtime: RuntimeRefLike | None = None) -> dict[str, ModelRef]:
    _ensure_model_profiles_registered()
    normalized_runtime = _runtime_name(runtime)
    names = sorted(_MODEL_REFS)
    supported: dict[str, ModelRef] = {}
    for normalized_name in names:
        model_ref = _MODEL_REFS[normalized_name]
        if normalized_runtime is not None and not _supports_runtime(
            normalized_name,
            normalized_runtime,
        ):
            continue
        supported[model_ref.name] = model_ref
    return supported


def resolve_model_alias(
    name: ModelRefLike | None,
    *,
    runtime: RuntimeRefLike | None = None,
) -> str | None:
    profile = resolve_model_profile(name, runtime=runtime)
    if profile is not None:
        return profile.model_id
    if not name:
        return None
    return name.name if isinstance(name, ModelRef) else name


def resolve_model_profile(
    name: ModelRefLike | None,
    *,
    runtime: RuntimeRefLike | None = None,
) -> TransformerModelProfile | None:
    if not name:
        return None
    _ensure_model_profiles_registered()
    normalized_name = normalize_name(name.name if isinstance(name, ModelRef) else name)
    normalized_runtime = _runtime_name(runtime)
    if normalized_runtime is not None:
        runtime_profile = _MODEL_PROFILES.get((normalized_runtime, normalized_name))
        if runtime_profile is not None:
            return runtime_profile
    shared_profile = _MODEL_PROFILES.get((None, normalized_name))
    if shared_profile is not None and (
        normalized_runtime is None or normalized_runtime in _SHARED_MODEL_RUNTIMES
    ):
        return shared_profile
    if normalized_runtime in {None, "transformers"}:
        return _MODEL_PROFILES.get(("transformers", normalized_name))
    return None


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


def _runtime_name(runtime: RuntimeRefLike | None) -> str | None:
    if runtime is None:
        return None
    return normalize_name(runtime.name if isinstance(runtime, RuntimeSpec) else runtime)


def _supports_runtime(name: str, runtime: str) -> bool:
    return (runtime, name) in _MODEL_PROFILES or (
        runtime in _SHARED_MODEL_RUNTIMES and (None, name) in _MODEL_PROFILES
    )

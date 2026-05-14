from __future__ import annotations

from aibackends.core.model_manager import ModelLocation, ModelManager
from aibackends.core.model_registry import resolve_model_profile
from aibackends.core.registry import ModelSupportSpec
from aibackends.core.types import RuntimeConfig


def ensure_model(
    manager: ModelManager,
    config: RuntimeConfig,
    resolved: str,
) -> ModelLocation:
    profile = resolve_model_profile(config.model, runtime=config.runtime)
    preferred_quantization = (
        profile.preferred_quantization if profile is not None else None
    )
    if preferred_quantization is None:
        preferred_quantization = config.extra_options.get(
            "preferred_quantization"
        )
    local_path = manager._download_gguf_repo(
        resolved,
        preferred_quantization=preferred_quantization,
    )
    return ModelLocation(source=resolved, local_path=str(local_path))


MODEL_SUPPORT_SPEC = ModelSupportSpec(
    runtime="llamacpp",
    ensure_model=ensure_model,
    pull_model=ensure_model,
)

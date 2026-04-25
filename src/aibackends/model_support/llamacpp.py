from __future__ import annotations

from aibackends.core.model_manager import ModelLocation, ModelManager
from aibackends.core.registry import ModelSupportSpec
from aibackends.core.types import RuntimeConfig


def ensure_model(
    manager: ModelManager,
    config: RuntimeConfig,
    resolved: str,
) -> ModelLocation:
    del config
    local_path = manager._download_gguf_repo(resolved)
    return ModelLocation(source=resolved, local_path=str(local_path))


MODEL_SUPPORT_SPEC = ModelSupportSpec(
    runtime="llamacpp",
    ensure_model=ensure_model,
    pull_model=ensure_model,
)

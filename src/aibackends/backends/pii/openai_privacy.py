from __future__ import annotations

import sys
import threading
from typing import Any

from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.core.registry import PIIBackendSpec
from aibackends.schemas.pii import PIIEntity

MODEL_ID = "openai/privacy-filter"

_PIPELINE_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def load_privacy_pipeline(spec: PIIBackendSpec) -> Any:
    """Build the token-classification pipeline once per process and cache it."""
    if spec.model_id is None:
        raise TaskExecutionError(f"The {spec.name} backend is missing a model_id.")

    cached = _PIPELINE_CACHE.get(spec.model_id)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _PIPELINE_CACHE.get(spec.model_id)
        if cached is not None:
            return cached
        if sys.version_info >= (3, 13):
            raise TaskExecutionError(
                "The openai/privacy-filter backend is not supported in this Python runtime."
            )
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[pii]' to use the openai/privacy-filter backend."
            ) from exc

        detector = pipeline(
            "token-classification",
            model=spec.model_id,
            aggregation_strategy="simple",
        )
        _PIPELINE_CACHE[spec.model_id] = detector
        return detector


def clear_pipeline_cache() -> None:
    """Drop any cached privacy-filter pipelines. Intended for tests."""
    with _CACHE_LOCK:
        _PIPELINE_CACHE.clear()


def detect_entities(
    spec: PIIBackendSpec,
    text: str,
    labels: list[str] | None,
) -> list[PIIEntity]:
    del labels
    detector = load_privacy_pipeline(spec)
    entities: list[PIIEntity] = []
    for item in detector(text):
        start = int(item.get("start", 0))
        end = int(item.get("end", start))
        entities.append(
            PIIEntity(
                entity_type=str(item.get("entity_group", "PII")).upper(),
                text=text[start:end],
                start=start,
                end=end,
                replacement="",
            )
        )
    return entities


PII_BACKEND_SPEC = PIIBackendSpec(
    name="openai-privacy",
    detect=detect_entities,
    aliases=("openai_privacy",),
    model_id=MODEL_ID,
    supports_custom_labels=False,
    metadata={"python_max": "3.12"},
    load_model=load_privacy_pipeline,
)

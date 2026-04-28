from __future__ import annotations

import threading
from typing import Any

from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.core.registry import PIIBackendSpec
from aibackends.schemas.pii import PIIEntity

GLINER_MODEL_ID = "nvidia/gliner-pii"
GLINER_LABELS = ("email", "phone_number", "user_name")
GLINER_THRESHOLD = 0.5

_MODEL_CACHE: dict[str, Any] = {}
_CACHE_LOCK = threading.Lock()


def load_gliner_model(spec: PIIBackendSpec) -> Any:
    """Load the GLiNER model for ``spec`` once per process and cache the instance."""
    if spec.model_id is None:
        raise TaskExecutionError(f"The {spec.name} backend is missing a model_id.")

    cached = _MODEL_CACHE.get(spec.model_id)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(spec.model_id)
        if cached is not None:
            return cached
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[pii]' to use the GLiNER PII backend."
            ) from exc
        model = GLiNER.from_pretrained(spec.model_id)
        _MODEL_CACHE[spec.model_id] = model
        return model


def detect_entities(
    spec: PIIBackendSpec,
    text: str,
    labels: list[str] | None,
) -> list[PIIEntity]:
    model = load_gliner_model(spec)
    selected_labels = list(labels) if labels else list(spec.default_labels)
    threshold = spec.threshold if spec.threshold is not None else GLINER_THRESHOLD
    raw_entities = model.predict_entities(text, selected_labels, threshold=threshold)

    entities: list[PIIEntity] = []
    for item in raw_entities:
        start = int(item.get("start", 0))
        end = int(item.get("end", start))
        if start < 0 or end <= start or end > len(text):
            continue
        entities.append(
            PIIEntity(
                entity_type=str(item.get("label", "PII")).upper().replace(" ", "_"),
                text=text[start:end],
                start=start,
                end=end,
                replacement="",
            )
        )
    return entities


def clear_model_cache() -> None:
    """Drop any cached GLiNER models. Intended for tests."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()


PII_BACKEND_SPEC = PIIBackendSpec(
    name="gliner",
    detect=detect_entities,
    model_id=GLINER_MODEL_ID,
    default_labels=GLINER_LABELS,
    threshold=GLINER_THRESHOLD,
    supports_custom_labels=True,
    load_model=load_gliner_model,
)

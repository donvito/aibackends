from __future__ import annotations

from threading import Lock
from typing import Any

from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.core.registry import PIIBackendSpec
from aibackends.schemas.pii import PIIEntity

GLINER_MODEL_ID = "nvidia/gliner-pii"
GLINER_LABELS = ("email", "phone_number", "user_name")
GLINER_THRESHOLD = 0.5
_MODEL_CACHE: dict[tuple[str, str | None], Any] = {}
_MODEL_CACHE_LOCK = Lock()


def detect_entities(
    spec: PIIBackendSpec,
    text: str,
    labels: list[str] | None,
    overrides: dict[str, Any] | None = None,
) -> list[PIIEntity]:
    raw_entities = _predict_raw_entities(spec, text, labels=labels, overrides=overrides)
    entities: list[PIIEntity] = []
    for item in raw_entities:
        start = int(item.get("start", 0))
        end = int(item.get("end", start))
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


def _predict_raw_entities(
    spec: PIIBackendSpec,
    text: str,
    *,
    labels: list[str] | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    try:
        model = _get_model(spec.model_id, device=_extract_device(overrides))
        raw_entities = model.predict_entities(
            text,
            labels or list(spec.default_labels),
            threshold=spec.threshold,
        )
    except RuntimeImportError:
        raise
    except Exception as exc:
        raise TaskExecutionError(f"GLiNER PII inference failed: {exc}") from exc
    return raw_entities if isinstance(raw_entities, list) else []


def _get_model(model_id: str | None, *, device: str | None = None) -> Any:
    if not model_id:
        raise TaskExecutionError("The GLiNER PII backend is missing a model ID.")
    cache_key = (model_id, device)
    with _MODEL_CACHE_LOCK:
        cached_model = _MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            return cached_model
        gliner_cls = _load_gliner_class()
        load_kwargs = {"map_location": device} if device else {}
        try:
            model = gliner_cls.from_pretrained(model_id, **load_kwargs)
        except Exception as exc:
            raise TaskExecutionError(f"Failed to load the GLiNER PII model '{model_id}': {exc}") from exc
        _MODEL_CACHE[cache_key] = model
        return model


def _extract_device(overrides: dict[str, Any] | None) -> str | None:
    if overrides is None:
        return None
    device = overrides.get("device")
    if device is None:
        return None
    return str(device)


def _load_gliner_class() -> Any:
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeImportError("Install 'aibackends[pii]' to use the GLiNER PII backend.") from exc
    return GLiNER


PII_BACKEND_SPEC = PIIBackendSpec(
    name="gliner",
    detect=detect_entities,
    model_id=GLINER_MODEL_ID,
    default_labels=GLINER_LABELS,
    threshold=GLINER_THRESHOLD,
    supports_custom_labels=True,
    metadata={"cache_strategy": "in_process_model"},
)

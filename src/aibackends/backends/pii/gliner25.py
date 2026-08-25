from __future__ import annotations

from typing import Any

from aibackends.backends.extraction.gliner25 import detect_pii_entities, load_extractor
from aibackends.backends.extraction.presets import PII_LABELS
from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import PIIBackendSpec
from aibackends.schemas.pii import PIIEntity

GLINER25_MODEL_ID = "fastino/gliner2.5-base-v1"
GLINER25_THRESHOLD = 0.5


def load_gliner25_model(spec: PIIBackendSpec) -> Any:
    """Load the GLiNER2.5 extractor used by the PII backend."""
    model_id = spec.model_id or GLINER25_MODEL_ID
    return load_extractor(device="cpu", model=model_id)


def detect_entities(
    spec: PIIBackendSpec,
    text: str,
    labels: list[str] | None,
) -> list[PIIEntity]:
    if spec.model_id is None:
        raise TaskExecutionError(f"The {spec.name} backend is missing a model_id.")
    selected_labels = list(labels) if labels else list(spec.default_labels)
    threshold = spec.threshold if spec.threshold is not None else GLINER25_THRESHOLD
    return detect_pii_entities(
        text,
        selected_labels,
        model=spec.model_id,
        threshold=threshold,
    )


PII_BACKEND_SPEC = PIIBackendSpec(
    name="gliner25",
    detect=detect_entities,
    aliases=("gliner2.5", "gliner-2.5"),
    model_id=GLINER25_MODEL_ID,
    default_labels=PII_LABELS,
    threshold=GLINER25_THRESHOLD,
    supports_custom_labels=True,
    load_model=load_gliner25_model,
)

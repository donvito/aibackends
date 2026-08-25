from __future__ import annotations

from typing import Any

from aibackends.backends.extraction.gliner25 import (
    DEFAULT_GLINER25_MODEL,
    DEFAULT_THRESHOLD,
    detect_pii_entities,
    load_gliner25_extractor,
)
from aibackends.core.registry import PIIBackendSpec
from aibackends.schemas.pii import PIIEntity

GLINER25_PII_LABELS = ("email", "phone_number", "person", "address")


def load_gliner25_pii_model(spec: PIIBackendSpec) -> Any:
    model_id = spec.model_id or DEFAULT_GLINER25_MODEL
    return load_gliner25_extractor(model_id, "cpu")


def detect_entities(
    spec: PIIBackendSpec,
    text: str,
    labels: list[str] | None,
) -> list[PIIEntity]:
    selected = list(labels) if labels else list(spec.default_labels)
    threshold = spec.threshold if spec.threshold is not None else DEFAULT_THRESHOLD
    return detect_pii_entities(
        text,
        selected,
        model_id=spec.model_id or DEFAULT_GLINER25_MODEL,
        threshold=threshold,
    )


PII_BACKEND_SPEC = PIIBackendSpec(
    name="gliner25",
    aliases=("gliner-2.5", "gliner2.5"),
    detect=detect_entities,
    model_id=DEFAULT_GLINER25_MODEL,
    default_labels=GLINER25_PII_LABELS,
    threshold=DEFAULT_THRESHOLD,
    supports_custom_labels=True,
    load_model=load_gliner25_pii_model,
)

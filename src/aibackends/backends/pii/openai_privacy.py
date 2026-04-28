from __future__ import annotations

import sys

from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.core.registry import PIIBackendSpec
from aibackends.schemas.pii import PIIEntity

MODEL_ID = "openai/privacy-filter"


def detect_entities(
    spec: PIIBackendSpec,
    text: str,
    labels: list[str] | None,
    overrides: dict[str, object] | None = None,
) -> list[PIIEntity]:
    del labels, overrides
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
)

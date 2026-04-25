from __future__ import annotations

from aibackends.core.types import AIBackendsModel


class PIIEntity(AIBackendsModel):
    entity_type: str
    text: str
    start: int
    end: int
    replacement: str


class RedactedText(AIBackendsModel):
    original_text: str
    redacted_text: str
    entities_found: list[PIIEntity]
    redaction_map: dict[str, str]
    backend_used: str


class Classification(AIBackendsModel):
    label: str
    confidence: float
    all_scores: dict[str, float]

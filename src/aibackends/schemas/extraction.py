from __future__ import annotations

from typing import Any

from pydantic import Field

from aibackends.core.types import AIBackendsModel


class SpanAttribute(AIBackendsModel):
    name: str
    label: str | list[str]
    confidence: float | None = None


class ExtractedEntity(AIBackendsModel):
    entity_type: str
    text: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None
    attributes: dict[str, SpanAttribute] = Field(default_factory=dict)


class EntityExtraction(AIBackendsModel):
    text: str
    entities: list[ExtractedEntity] = Field(default_factory=list)
    backend_used: str
    model_id: str


class RecordExtraction(AIBackendsModel):
    text: str
    records: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    backend_used: str
    model_id: str


class GraphEntity(AIBackendsModel):
    id: str
    entity_type: str
    text: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None


class GraphRelation(AIBackendsModel):
    relation_type: str
    head_id: str
    tail_id: str
    head_text: str
    tail_text: str
    confidence: float | None = None


class GraphExtraction(AIBackendsModel):
    text: str
    entities: list[GraphEntity] = Field(default_factory=list)
    relations: list[GraphRelation] = Field(default_factory=list)
    feasible: bool = True
    backend_used: str
    model_id: str


class TaskClassification(AIBackendsModel):
    task: str
    value: str | list[str]
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class SchemaClassification(AIBackendsModel):
    text: str
    tasks: dict[str, TaskClassification] = Field(default_factory=dict)
    feasible: bool = True
    backend_used: str
    model_id: str

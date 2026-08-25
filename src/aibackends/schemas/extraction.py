from __future__ import annotations

from pydantic import Field

from aibackends.core.types import AIBackendsModel


class SpanAttribute(AIBackendsModel):
    """Attribute values decoded at an extracted span for one attribute group."""

    labels: list[str]
    confidences: dict[str, float] = Field(default_factory=dict)

    @property
    def label(self) -> str | None:
        """Top attribute label, or None when nothing cleared the threshold."""
        return self.labels[0] if self.labels else None


class ExtractedEntity(AIBackendsModel):
    """One extracted span with half-open character offsets into the source text."""

    label: str
    text: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None
    attributes: dict[str, SpanAttribute] = Field(default_factory=dict)


class EntityExtraction(AIBackendsModel):
    text: str
    entities: list[ExtractedEntity]
    backend_used: str
    model_id: str

    def by_label(self, label: str) -> list[ExtractedEntity]:
        return [entity for entity in self.entities if entity.label == label]


class TaskClassification(AIBackendsModel):
    """Decoded labels for one classification task."""

    task: str
    labels: list[str]
    multi_label: bool = False
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class TextClassification(AIBackendsModel):
    text: str
    tasks: dict[str, TaskClassification]
    feasible: bool = True
    constrained: bool = False
    backend_used: str
    model_id: str

    def value(self, task: str) -> str | None:
        """Single selected label for a task, or None when nothing was selected."""
        labels = self.tasks[task].labels
        return labels[0] if labels else None

    def values(self, task: str) -> list[str]:
        return list(self.tasks[task].labels)


class GraphEntity(AIBackendsModel):
    id: str
    type: str
    text: str
    start: int | None = None
    end: int | None = None
    confidence: float | None = None


class GraphRelation(AIBackendsModel):
    type: str
    head: str
    tail: str
    head_text: str
    tail_text: str
    confidence: float | None = None


class KnowledgeGraph(AIBackendsModel):
    text: str
    entities: list[GraphEntity]
    relations: list[GraphRelation]
    feasible: bool = True
    backend_used: str
    model_id: str

    def entity(self, entity_id: str) -> GraphEntity:
        for candidate in self.entities:
            if candidate.id == entity_id:
                return candidate
        raise KeyError(entity_id)

    def triples(self) -> list[tuple[str, str, str]]:
        """Relations as (head text, relation type, tail text) triples."""
        return [
            (relation.head_text, relation.type, relation.tail_text)
            for relation in self.relations
        ]

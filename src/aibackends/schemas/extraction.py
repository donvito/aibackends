from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from aibackends.core.types import AIBackendsModel

ConstraintKind = Literal["implies", "excludes"]


class SpanAttribute(AIBackendsModel):
    label: str
    confidence: float | None = None


class ExtractedEntity(AIBackendsModel):
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float | None = None
    attributes: dict[str, SpanAttribute] = Field(default_factory=dict)


class EntityExtraction(AIBackendsModel):
    text: str
    entities: list[ExtractedEntity]
    backend_used: str
    model_id: str
    long_document: bool = False


class ExtractedRelation(AIBackendsModel):
    relation_type: str
    head: ExtractedEntity
    tail: ExtractedEntity
    confidence: float | None = None


class RelationExtraction(AIBackendsModel):
    text: str
    relations: list[ExtractedRelation]
    backend_used: str
    model_id: str


class GraphEntity(AIBackendsModel):
    id: str
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float | None = None


class GraphRelation(AIBackendsModel):
    relation_type: str
    head: str
    tail: str
    confidence: float | None = None


class KnowledgeGraph(AIBackendsModel):
    text: str
    entities: list[GraphEntity]
    relations: list[GraphRelation]
    feasible: bool
    backend_used: str
    model_id: str
    long_document: bool = False


class ClassificationTaskSpec(AIBackendsModel):
    name: str
    labels: list[str]
    multi_label: bool = False
    min_labels: int | None = None
    max_labels: int | None = None


class ClassificationConstraint(AIBackendsModel):
    kind: ConstraintKind
    source: tuple[str, str]
    target: tuple[str, str]


class ConstrainedClassificationSchema(AIBackendsModel):
    tasks: list[ClassificationTaskSpec]
    constraints: list[ClassificationConstraint] = Field(default_factory=list)


class ClassificationTaskResult(AIBackendsModel):
    task: str
    value: str | list[str]
    confidence: float | None = None
    probabilities: dict[str, float] = Field(default_factory=dict)


class ConstrainedClassification(AIBackendsModel):
    text: str
    tasks: list[ClassificationTaskResult]
    feasible: bool
    backend_used: str
    model_id: str

    def value(self, task: str) -> str | list[str] | None:
        for item in self.tasks:
            if item.task == task:
                return item.value
        return None


class RelationSpec(AIBackendsModel):
    name: str
    head_type: str
    tail_type: str
    unique_head: bool = False
    unique_tail: bool = False


class AttributeSpec(AIBackendsModel):
    name: str
    labels: list[str]
    applies_to: list[str] | None = None
    qualify_labels: bool = True


class RoutingDecision(AIBackendsModel):
    text: str
    intent: str
    destination: str
    feasible: bool
    classification: ConstrainedClassification
    backend_used: str
    model_id: str


class AgentGuardrail(AIBackendsModel):
    text: str
    safety: str
    harm_type: str | None = None
    is_allowed: bool
    feasible: bool
    classification: ConstrainedClassification
    backend_used: str
    model_id: str


class ClinicalMention(AIBackendsModel):
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float | None = None
    negation: str | None = None
    dosage_form: str | None = None


class ClinicalExtraction(AIBackendsModel):
    text: str
    mentions: list[ClinicalMention]
    backend_used: str
    model_id: str


class ContractReview(AIBackendsModel):
    text: str
    parties: list[ExtractedEntity] = Field(default_factory=list)
    obligations: list[ExtractedEntity] = Field(default_factory=list)
    termination_clauses: list[ExtractedEntity] = Field(default_factory=list)
    dates: list[ExtractedEntity] = Field(default_factory=list)
    amounts: list[ExtractedEntity] = Field(default_factory=list)
    addresses: list[ExtractedEntity] = Field(default_factory=list)
    backend_used: str
    model_id: str
    long_document: bool = True


def coerce_task_specs(
    tasks: list[ClassificationTaskSpec] | list[dict[str, Any]],
) -> list[ClassificationTaskSpec]:
    resolved: list[ClassificationTaskSpec] = []
    for task in tasks:
        if isinstance(task, ClassificationTaskSpec):
            resolved.append(task)
        else:
            resolved.append(ClassificationTaskSpec.model_validate(task))
    return resolved


def coerce_constraints(
    constraints: list[ClassificationConstraint] | list[dict[str, Any]] | None,
) -> list[ClassificationConstraint]:
    if not constraints:
        return []
    resolved: list[ClassificationConstraint] = []
    for item in constraints:
        if isinstance(item, ClassificationConstraint):
            resolved.append(item)
        else:
            resolved.append(ClassificationConstraint.model_validate(item))
    return resolved


def coerce_relation_specs(
    relations: list[RelationSpec] | list[dict[str, Any]],
) -> list[RelationSpec]:
    resolved: list[RelationSpec] = []
    for item in relations:
        if isinstance(item, RelationSpec):
            resolved.append(item)
        else:
            resolved.append(RelationSpec.model_validate(item))
    return resolved


def coerce_attribute_specs(
    attributes: list[AttributeSpec] | list[dict[str, Any]],
) -> list[AttributeSpec]:
    resolved: list[AttributeSpec] = []
    for item in attributes:
        if isinstance(item, AttributeSpec):
            resolved.append(item)
        else:
            resolved.append(AttributeSpec.model_validate(item))
    return resolved

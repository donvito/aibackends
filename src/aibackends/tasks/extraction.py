from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aibackends.backends.extraction import get_extraction_backend
from aibackends.backends.extraction.presets import (
    CLINICAL_ATTRIBUTES,
    CLINICAL_LABELS,
    CONTRACT_LABELS,
    GUARDRAIL_SCHEMA,
    MEMORY_GRAPH_ENTITIES,
    MEMORY_GRAPH_RELATIONS,
    ROUTING_SCHEMA,
)
from aibackends.core.registry import TaskSpec
from aibackends.schemas.extraction import (
    AgentGuardrail,
    AttributeSpec,
    ClassificationConstraint,
    ClassificationTaskSpec,
    ClinicalExtraction,
    ClinicalMention,
    ConstrainedClassification,
    ContractReview,
    EntityExtraction,
    ExtractedEntity,
    KnowledgeGraph,
    RelationExtraction,
    RelationSpec,
    RoutingDecision,
    coerce_attribute_specs,
    coerce_constraints,
    coerce_relation_specs,
    coerce_task_specs,
)
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import load_text_input

DEFAULT_BACKEND = "gliner25"
DEFAULT_DEVICE = "cpu"
DEFAULT_THRESHOLD = 0.5


def _common_options(
    *,
    device: str,
    model: str | None,
    threshold: float | None = None,
    long_document: bool | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "device": device,
        "model": model,
    }
    if threshold is not None:
        options["threshold"] = threshold
    if long_document is not None:
        options["long_document"] = long_document
    if chunk_size is not None:
        options["chunk_size"] = chunk_size
    if chunk_overlap is not None:
        options["chunk_overlap"] = chunk_overlap
    return options


def extract_entities(
    text: str | Path,
    labels: list[str] | dict[str, str],
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    long_document: bool | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> EntityExtraction:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_entities(
        content,
        labels,
        **_common_options(
            device=device,
            model=model,
            threshold=threshold,
            long_document=long_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ),
    )


async def extract_entities_async(
    text: str | Path,
    labels: list[str] | dict[str, str],
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    long_document: bool | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> EntityExtraction:
    return await asyncio.to_thread(
        extract_entities,
        text,
        labels,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def extract_relations(
    text: str | Path,
    relations: list[str],
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> RelationExtraction:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_relations(
        content,
        relations,
        device=device,
        model=model,
        threshold=threshold,
    )


async def extract_relations_async(
    text: str | Path,
    relations: list[str],
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> RelationExtraction:
    return await asyncio.to_thread(
        extract_relations,
        text,
        relations,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
    )


def extract_graph(
    text: str | Path,
    *,
    entities: list[str],
    relations: list[RelationSpec] | list[dict[str, Any]],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    no_self_loops: bool = True,
    long_document: bool | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    beam_size: int = 32,
) -> KnowledgeGraph:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_graph(
        content,
        list(entities),
        coerce_relation_specs(relations),
        device=device,
        model=model,
        no_self_loops=no_self_loops,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        beam_size=beam_size,
    )


async def extract_graph_async(
    text: str | Path,
    *,
    entities: list[str],
    relations: list[RelationSpec] | list[dict[str, Any]],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    no_self_loops: bool = True,
    long_document: bool | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    beam_size: int = 32,
) -> KnowledgeGraph:
    return await asyncio.to_thread(
        extract_graph,
        text,
        entities=entities,
        relations=relations,
        backend=backend,
        device=device,
        model=model,
        no_self_loops=no_self_loops,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        beam_size=beam_size,
    )


def classify_constrained(
    text: str | Path,
    *,
    tasks: list[ClassificationTaskSpec] | list[dict[str, Any]],
    constraints: list[ClassificationConstraint] | list[dict[str, Any]] | None = None,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    long_document: bool | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    decoder: str = "exact",
    beam_size: int = 16,
) -> ConstrainedClassification:
    content = load_text_input(text)
    return get_extraction_backend(backend).classify_constrained(
        content,
        coerce_task_specs(tasks),
        coerce_constraints(constraints),
        device=device,
        model=model,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        decoder=decoder,
        beam_size=beam_size,
    )


async def classify_constrained_async(
    text: str | Path,
    *,
    tasks: list[ClassificationTaskSpec] | list[dict[str, Any]],
    constraints: list[ClassificationConstraint] | list[dict[str, Any]] | None = None,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    long_document: bool | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    decoder: str = "exact",
    beam_size: int = 16,
) -> ConstrainedClassification:
    return await asyncio.to_thread(
        classify_constrained,
        text,
        tasks=tasks,
        constraints=constraints,
        backend=backend,
        device=device,
        model=model,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        decoder=decoder,
        beam_size=beam_size,
    )


def extract_span_attributes(
    text: str | Path,
    labels: list[str] | dict[str, str],
    attributes: list[AttributeSpec] | list[dict[str, Any]],
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    long_document: bool | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> EntityExtraction:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_with_attributes(
        content,
        labels,
        coerce_attribute_specs(attributes),
        device=device,
        model=model,
        threshold=threshold,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def extract_span_attributes_async(
    text: str | Path,
    labels: list[str] | dict[str, str],
    attributes: list[AttributeSpec] | list[dict[str, Any]],
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    long_document: bool | None = None,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> EntityExtraction:
    return await asyncio.to_thread(
        extract_span_attributes,
        text,
        labels,
        attributes,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _task_value(result: ConstrainedClassification, name: str) -> str:
    value = result.value(name)
    if isinstance(value, list):
        return str(value[0]) if value else ""
    if isinstance(value, str):
        return value
    return ""


def route_agent(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
) -> RoutingDecision:
    """Route a request to a model tier or sub-agent under compatibility rules."""
    classification = classify_constrained(
        text,
        tasks=list(ROUTING_SCHEMA.tasks),
        constraints=list(ROUTING_SCHEMA.constraints),
        backend=backend,
        device=device,
        model=model,
    )
    return RoutingDecision(
        text=classification.text,
        intent=_task_value(classification, "intent") or "chat",
        destination=_task_value(classification, "destination") or "small_model",
        feasible=classification.feasible,
        classification=classification,
        backend_used=classification.backend_used,
        model_id=classification.model_id,
    )


async def route_agent_async(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
) -> RoutingDecision:
    return await asyncio.to_thread(
        route_agent,
        text,
        backend=backend,
        device=device,
        model=model,
    )


def screen_agent_action(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
) -> AgentGuardrail:
    """Screen an agent action so harm types cannot appear on an allow verdict."""
    classification = classify_constrained(
        text,
        tasks=list(GUARDRAIL_SCHEMA.tasks),
        constraints=list(GUARDRAIL_SCHEMA.constraints),
        backend=backend,
        device=device,
        model=model,
    )
    safety = _task_value(classification, "safety") or "block"
    harm = _task_value(classification, "harm_type")
    harm_type = None if not harm or harm == "none" else harm
    return AgentGuardrail(
        text=classification.text,
        safety=safety,
        harm_type=harm_type,
        is_allowed=safety == "allow" and classification.feasible,
        feasible=classification.feasible,
        classification=classification,
        backend_used=classification.backend_used,
        model_id=classification.model_id,
    )


async def screen_agent_action_async(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
) -> AgentGuardrail:
    return await asyncio.to_thread(
        screen_agent_action,
        text,
        backend=backend,
        device=device,
        model=model,
    )


def extract_memory_graph(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    long_document: bool | None = None,
) -> KnowledgeGraph:
    """Build a typed people/projects/commitments graph for agent memory."""
    return extract_graph(
        text,
        entities=list(MEMORY_GRAPH_ENTITIES),
        relations=list(MEMORY_GRAPH_RELATIONS),
        backend=backend,
        device=device,
        model=model,
        long_document=long_document,
    )


async def extract_memory_graph_async(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    long_document: bool | None = None,
) -> KnowledgeGraph:
    return await asyncio.to_thread(
        extract_memory_graph,
        text,
        backend=backend,
        device=device,
        model=model,
        long_document=long_document,
    )


def review_contract(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    long_document: bool | None = True,
) -> ContractReview:
    """Extract parties, obligations, and termination language from a contract."""
    extraction = extract_entities(
        text,
        CONTRACT_LABELS,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        long_document=long_document,
    )
    grouped: dict[str, list[ExtractedEntity]] = {
        "party": [],
        "obligation": [],
        "termination_clause": [],
        "date": [],
        "amount": [],
        "address": [],
    }
    for entity in extraction.entities:
        if entity.entity_type in grouped:
            grouped[entity.entity_type].append(entity)
    return ContractReview(
        text=extraction.text,
        parties=grouped["party"],
        obligations=grouped["obligation"],
        termination_clauses=grouped["termination_clause"],
        dates=grouped["date"],
        amounts=grouped["amount"],
        addresses=grouped["address"],
        backend_used=extraction.backend_used,
        model_id=extraction.model_id,
        long_document=extraction.long_document,
    )


async def review_contract_async(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    long_document: bool | None = True,
) -> ContractReview:
    return await asyncio.to_thread(
        review_contract,
        text,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        long_document=long_document,
    )


def extract_clinical(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> ClinicalExtraction:
    """Extract symptoms and medications with negation and dosage form."""
    extraction = extract_span_attributes(
        text,
        CLINICAL_LABELS,
        CLINICAL_ATTRIBUTES,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
    )
    mentions: list[ClinicalMention] = []
    for entity in extraction.entities:
        negation = None
        if "negation" in entity.attributes:
            negation = entity.attributes["negation"].label
        dosage_form = None
        if "dosage_form" in entity.attributes:
            dosage_form = entity.attributes["dosage_form"].label
        mentions.append(
            ClinicalMention(
                entity_type=entity.entity_type,
                text=entity.text,
                start=entity.start,
                end=entity.end,
                confidence=entity.confidence,
                negation=negation,
                dosage_form=dosage_form,
            )
        )
    return ClinicalExtraction(
        text=extraction.text,
        mentions=mentions,
        backend_used=extraction.backend_used,
        model_id=extraction.model_id,
    )


async def extract_clinical_async(
    text: str | Path,
    *,
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> ClinicalExtraction:
    return await asyncio.to_thread(
        extract_clinical,
        text,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
    )


class ExtractEntitiesTask(BaseTask):
    name = "extract-entities"

    def run(
        self,
        input: str | Path,
        *,
        labels: list[str] | dict[str, str] | None = None,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        threshold: float | None = None,
        long_document: bool | None = None,
        **overrides: Any,
    ) -> EntityExtraction:
        options = self._resolve_options(
            labels=labels,
            backend=backend,
            device=device,
            model=model,
            threshold=threshold,
            long_document=long_document,
            **overrides,
        )
        selected_labels = options.pop("labels", None)
        if not selected_labels:
            raise ValueError("extract-entities requires labels.")
        options.setdefault("backend", DEFAULT_BACKEND)
        options.setdefault("device", DEFAULT_DEVICE)
        return extract_entities(input, selected_labels, **options)


class RouteAgentTask(BaseTask):
    name = "route-agent"

    def run(
        self,
        input: str | Path,
        *,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        **overrides: Any,
    ) -> RoutingDecision:
        options = self._resolve_options(
            backend=backend,
            device=device,
            model=model,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        options.setdefault("device", DEFAULT_DEVICE)
        return route_agent(input, **options)


class ScreenAgentActionTask(BaseTask):
    name = "screen-agent-action"

    def run(
        self,
        input: str | Path,
        *,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        **overrides: Any,
    ) -> AgentGuardrail:
        options = self._resolve_options(
            backend=backend,
            device=device,
            model=model,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        options.setdefault("device", DEFAULT_DEVICE)
        return screen_agent_action(input, **options)


class ExtractMemoryGraphTask(BaseTask):
    name = "extract-memory-graph"

    def run(
        self,
        input: str | Path,
        *,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        long_document: bool | None = None,
        **overrides: Any,
    ) -> KnowledgeGraph:
        options = self._resolve_options(
            backend=backend,
            device=device,
            model=model,
            long_document=long_document,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        options.setdefault("device", DEFAULT_DEVICE)
        return extract_memory_graph(input, **options)


class ReviewContractTask(BaseTask):
    name = "review-contract"

    def run(
        self,
        input: str | Path,
        *,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        threshold: float | None = None,
        long_document: bool | None = None,
        **overrides: Any,
    ) -> ContractReview:
        options = self._resolve_options(
            backend=backend,
            device=device,
            model=model,
            threshold=threshold,
            long_document=long_document,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        options.setdefault("device", DEFAULT_DEVICE)
        return review_contract(input, **options)


class ExtractClinicalTask(BaseTask):
    name = "extract-clinical"

    def run(
        self,
        input: str | Path,
        *,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        threshold: float | None = None,
        **overrides: Any,
    ) -> ClinicalExtraction:
        options = self._resolve_options(
            backend=backend,
            device=device,
            model=model,
            threshold=threshold,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        options.setdefault("device", DEFAULT_DEVICE)
        return extract_clinical(input, **options)


TASK_SPEC: Sequence[TaskSpec] = (
    TaskSpec(
        name=ExtractEntitiesTask.name,
        task_factory=ExtractEntitiesTask,
        aliases=("extract_entities",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_model_id=True,
        accepts_backend=True,
        accepts_labels=True,
        accepts_device=True,
        accepts_threshold=True,
        requires_labels=True,
    ),
    TaskSpec(
        name=RouteAgentTask.name,
        task_factory=RouteAgentTask,
        aliases=("route_agent",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_model_id=True,
        accepts_backend=True,
        accepts_device=True,
    ),
    TaskSpec(
        name=ScreenAgentActionTask.name,
        task_factory=ScreenAgentActionTask,
        aliases=("screen_agent_action",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_model_id=True,
        accepts_backend=True,
        accepts_device=True,
    ),
    TaskSpec(
        name=ExtractMemoryGraphTask.name,
        task_factory=ExtractMemoryGraphTask,
        aliases=("extract_memory_graph",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_model_id=True,
        accepts_backend=True,
        accepts_device=True,
    ),
    TaskSpec(
        name=ReviewContractTask.name,
        task_factory=ReviewContractTask,
        aliases=("review_contract",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_model_id=True,
        accepts_backend=True,
        accepts_device=True,
        accepts_threshold=True,
    ),
    TaskSpec(
        name=ExtractClinicalTask.name,
        task_factory=ExtractClinicalTask,
        aliases=("extract_clinical",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_model_id=True,
        accepts_backend=True,
        accepts_device=True,
        accepts_threshold=True,
    ),
)

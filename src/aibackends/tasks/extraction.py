from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aibackends.backends.extraction import get_extraction_backend
from aibackends.backends.extraction._base import (
    AttributesInput,
    ConstraintsInput,
    LabelsInput,
    RelationsInput,
    TasksInput,
)
from aibackends.core.registry import ModelRef, TaskSpec
from aibackends.schemas.extraction import (
    EntityExtraction,
    KnowledgeGraph,
    TextClassification,
)
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import load_text_input

DEFAULT_BACKEND = "gliner2.5"


def extract_entities(
    text: str | Path,
    *,
    labels: LabelsInput,
    attributes: AttributesInput | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
    long_document: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> EntityExtraction:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_entities(
        content,
        labels=labels,
        attributes=attributes,
        model=model,
        device=device,
        threshold=threshold,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def extract_entities_async(
    text: str | Path,
    *,
    labels: LabelsInput,
    attributes: AttributesInput | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
    long_document: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> EntityExtraction:
    return await asyncio.to_thread(
        extract_entities,
        text,
        labels=labels,
        attributes=attributes,
        backend=backend,
        model=model,
        device=device,
        threshold=threshold,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def extract_entities_batch(
    texts: Sequence[str | Path],
    *,
    labels: LabelsInput,
    attributes: AttributesInput | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
    batch_size: int = 8,
    long_document: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> list[EntityExtraction]:
    contents = [load_text_input(text) for text in texts]
    return get_extraction_backend(backend).extract_entities_batch(
        contents,
        labels=labels,
        attributes=attributes,
        model=model,
        device=device,
        threshold=threshold,
        batch_size=batch_size,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def extract_entities_batch_async(
    texts: Sequence[str | Path],
    *,
    labels: LabelsInput,
    attributes: AttributesInput | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
    threshold: float = 0.5,
    batch_size: int = 8,
    long_document: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> list[EntityExtraction]:
    return await asyncio.to_thread(
        extract_entities_batch,
        texts,
        labels=labels,
        attributes=attributes,
        backend=backend,
        model=model,
        device=device,
        threshold=threshold,
        batch_size=batch_size,
        long_document=long_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _resolve_classification_tasks(
    tasks: TasksInput | None,
    labels: LabelsInput | None,
) -> TasksInput:
    if tasks is not None:
        return tasks
    if labels is not None:
        label_list = list(labels.keys()) if isinstance(labels, dict) else list(labels)
        return {"label": {"labels": label_list}}
    raise ValueError("classify_text requires either 'tasks' or 'labels'.")


def classify_text(
    text: str | Path,
    *,
    tasks: TasksInput | None = None,
    labels: LabelsInput | None = None,
    constraints: ConstraintsInput | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
) -> TextClassification:
    content = load_text_input(text)
    return get_extraction_backend(backend).classify_text(
        content,
        tasks=_resolve_classification_tasks(tasks, labels),
        constraints=constraints,
        model=model,
        device=device,
    )


async def classify_text_async(
    text: str | Path,
    *,
    tasks: TasksInput | None = None,
    labels: LabelsInput | None = None,
    constraints: ConstraintsInput | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
) -> TextClassification:
    return await asyncio.to_thread(
        classify_text,
        text,
        tasks=tasks,
        labels=labels,
        constraints=constraints,
        backend=backend,
        model=model,
        device=device,
    )


def classify_texts(
    texts: Sequence[str | Path],
    *,
    tasks: TasksInput | None = None,
    labels: LabelsInput | None = None,
    constraints: ConstraintsInput | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
    batch_size: int = 8,
) -> list[TextClassification]:
    contents = [load_text_input(text) for text in texts]
    return get_extraction_backend(backend).classify_text_batch(
        contents,
        tasks=_resolve_classification_tasks(tasks, labels),
        constraints=constraints,
        model=model,
        device=device,
        batch_size=batch_size,
    )


async def classify_texts_async(
    texts: Sequence[str | Path],
    *,
    tasks: TasksInput | None = None,
    labels: LabelsInput | None = None,
    constraints: ConstraintsInput | None = None,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
    batch_size: int = 8,
) -> list[TextClassification]:
    return await asyncio.to_thread(
        classify_texts,
        texts,
        tasks=tasks,
        labels=labels,
        constraints=constraints,
        backend=backend,
        model=model,
        device=device,
        batch_size=batch_size,
    )


def extract_graph(
    text: str | Path,
    *,
    entities: LabelsInput,
    relations: RelationsInput,
    no_self_loops: bool = True,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
    optimizer: str = "beam",
    beam_size: int = 32,
) -> KnowledgeGraph:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_graph(
        content,
        entities=entities,
        relations=relations,
        no_self_loops=no_self_loops,
        model=model,
        device=device,
        optimizer=optimizer,
        beam_size=beam_size,
    )


async def extract_graph_async(
    text: str | Path,
    *,
    entities: LabelsInput,
    relations: RelationsInput,
    no_self_loops: bool = True,
    backend: str = DEFAULT_BACKEND,
    model: str | None = None,
    device: str = "cpu",
    optimizer: str = "beam",
    beam_size: int = 32,
) -> KnowledgeGraph:
    return await asyncio.to_thread(
        extract_graph,
        text,
        entities=entities,
        relations=relations,
        no_self_loops=no_self_loops,
        backend=backend,
        model=model,
        device=device,
        optimizer=optimizer,
        beam_size=beam_size,
    )


class _ExtractionTask(BaseTask):
    """Base task that accepts model variants as plain strings or ModelRef values."""

    def __init__(self, **defaults: Any) -> None:
        _coerce_model_ref(defaults)
        super().__init__(**defaults)

    def _resolve_options(self, **overrides: Any) -> dict[str, Any]:
        options = dict(self.defaults)
        options.update({key: value for key, value in overrides.items() if value is not None})
        options.pop("runtime", None)
        model = options.get("model")
        if isinstance(model, ModelRef):
            options["model"] = model.name
        return options


def _coerce_model_ref(values: dict[str, Any]) -> None:
    model = values.get("model")
    if isinstance(model, str):
        values["model"] = ModelRef(name=model)


class ExtractEntitiesTask(_ExtractionTask):
    name = "extract-entities"

    def run(
        self,
        input: str | Path,
        *,
        labels: LabelsInput | None = None,
        attributes: AttributesInput | None = None,
        backend: str | None = None,
        model: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        long_document: bool | None = None,
        **overrides: Any,
    ) -> EntityExtraction:
        options = self._resolve_options(
            labels=labels,
            attributes=attributes,
            backend=backend,
            model=model,
            device=device,
            threshold=threshold,
            long_document=long_document,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        if "labels" not in options:
            raise ValueError("extract-entities requires 'labels'.")
        return extract_entities(input, **options)

    def run_batch(
        self,
        inputs: Sequence[str | Path],
        *,
        labels: LabelsInput | None = None,
        attributes: AttributesInput | None = None,
        backend: str | None = None,
        model: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        batch_size: int | None = None,
        long_document: bool | None = None,
        **overrides: Any,
    ) -> list[EntityExtraction]:
        options = self._resolve_options(
            labels=labels,
            attributes=attributes,
            backend=backend,
            model=model,
            device=device,
            threshold=threshold,
            batch_size=batch_size,
            long_document=long_document,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        if "labels" not in options:
            raise ValueError("extract-entities requires 'labels'.")
        return extract_entities_batch(inputs, **options)


class ClassifyTextTask(_ExtractionTask):
    name = "classify-text"

    def run(
        self,
        input: str | Path,
        *,
        tasks: TasksInput | None = None,
        labels: LabelsInput | None = None,
        constraints: ConstraintsInput | None = None,
        backend: str | None = None,
        model: str | None = None,
        device: str | None = None,
        **overrides: Any,
    ) -> TextClassification:
        options = self._resolve_options(
            tasks=tasks,
            labels=labels,
            constraints=constraints,
            backend=backend,
            model=model,
            device=device,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        return classify_text(input, **options)

    def run_batch(
        self,
        inputs: Sequence[str | Path],
        *,
        tasks: TasksInput | None = None,
        labels: LabelsInput | None = None,
        constraints: ConstraintsInput | None = None,
        backend: str | None = None,
        model: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        **overrides: Any,
    ) -> list[TextClassification]:
        options = self._resolve_options(
            tasks=tasks,
            labels=labels,
            constraints=constraints,
            backend=backend,
            model=model,
            device=device,
            batch_size=batch_size,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        return classify_texts(inputs, **options)


class ExtractGraphTask(_ExtractionTask):
    name = "extract-graph"

    def run(
        self,
        input: str | Path,
        *,
        entities: LabelsInput | None = None,
        relations: RelationsInput | None = None,
        no_self_loops: bool | None = None,
        backend: str | None = None,
        model: str | None = None,
        device: str | None = None,
        **overrides: Any,
    ) -> KnowledgeGraph:
        options = self._resolve_options(
            entities=entities,
            relations=relations,
            no_self_loops=no_self_loops,
            backend=backend,
            model=model,
            device=device,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_BACKEND)
        if "entities" not in options or "relations" not in options:
            raise ValueError("extract-graph requires 'entities' and 'relations'.")
        return extract_graph(input, **options)


TASK_SPEC = (
    TaskSpec(
        name=ExtractEntitiesTask.name,
        task_factory=ExtractEntitiesTask,
        aliases=("extract_entities",),
        accepts_runtime=False,
        accepts_model=True,
        accepts_backend=True,
        accepts_labels=True,
        accepts_device=True,
        accepts_threshold=True,
        requires_labels=True,
    ),
    TaskSpec(
        name=ClassifyTextTask.name,
        task_factory=ClassifyTextTask,
        aliases=("classify_text",),
        accepts_runtime=False,
        accepts_model=True,
        accepts_backend=True,
        accepts_labels=True,
        accepts_device=True,
        requires_labels=True,
    ),
    TaskSpec(
        name=ExtractGraphTask.name,
        task_factory=ExtractGraphTask,
        aliases=("extract_graph",),
        accepts_runtime=False,
        accepts_model=True,
        accepts_backend=True,
        accepts_device=True,
        accepts_entities=True,
        accepts_relations=True,
    ),
)

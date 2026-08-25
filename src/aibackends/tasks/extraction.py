from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aibackends.backends.extraction import get_extraction_backend
from aibackends.core.registry import TaskSpec
from aibackends.schemas.extraction import (
    EntityExtraction,
    GraphExtraction,
    RecordExtraction,
    SchemaClassification,
)
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import load_text_input

DEFAULT_BACKEND = "gliner25"
DEFAULT_MODEL = "gliner25-small"
DEFAULT_DEVICE = "cpu"


def extract_entities(
    text: str | Path,
    *,
    labels: Sequence[str] | Mapping[str, str],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    long: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    attributes: Mapping[str, Any] | None = None,
) -> EntityExtraction:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_entities(
        content,
        labels,
        device=device,
        model=model,
        threshold=threshold,
        long=long,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        attributes=attributes,
    )


async def extract_entities_async(
    text: str | Path,
    *,
    labels: Sequence[str] | Mapping[str, str],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    long: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    attributes: Mapping[str, Any] | None = None,
) -> EntityExtraction:
    return await asyncio.to_thread(
        extract_entities,
        text,
        labels=labels,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        long=long,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        attributes=attributes,
    )


def extract_entities_long(
    text: str | Path,
    *,
    labels: Sequence[str] | Mapping[str, str],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    attributes: Mapping[str, Any] | None = None,
) -> EntityExtraction:
    return extract_entities(
        text,
        labels=labels,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        long=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        attributes=attributes,
    )


def extract_entities_batch(
    texts: Sequence[str | Path],
    *,
    labels: Sequence[str] | Mapping[str, str],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    long: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    attributes: Mapping[str, Any] | None = None,
    batch_size: int = 8,
) -> list[EntityExtraction]:
    contents = [load_text_input(text) for text in texts]
    return get_extraction_backend(backend).extract_entities_batch(
        contents,
        labels,
        device=device,
        model=model,
        threshold=threshold,
        long=long,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        attributes=attributes,
        batch_size=batch_size,
    )


async def extract_entities_batch_async(
    texts: Sequence[str | Path],
    *,
    labels: Sequence[str] | Mapping[str, str],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    long: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    attributes: Mapping[str, Any] | None = None,
    batch_size: int = 8,
) -> list[EntityExtraction]:
    return await asyncio.to_thread(
        extract_entities_batch,
        texts,
        labels=labels,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        long=long,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        attributes=attributes,
        batch_size=batch_size,
    )


def extract_records(
    text: str | Path,
    *,
    schema: Mapping[str, Any],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    long: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> RecordExtraction:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_records(
        content,
        schema,
        device=device,
        model=model,
        threshold=threshold,
        long=long,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def extract_records_async(
    text: str | Path,
    *,
    schema: Mapping[str, Any],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    long: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> RecordExtraction:
    return await asyncio.to_thread(
        extract_records,
        text,
        schema=schema,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        long=long,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def classify_schema(
    text: str | Path,
    *,
    tasks: Mapping[str, Mapping[str, Any] | Sequence[str]],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    constraints: Sequence[Mapping[str, Any]] | None = None,
) -> SchemaClassification:
    content = load_text_input(text)
    return get_extraction_backend(backend).classify_schema(
        content,
        tasks,
        device=device,
        model=model,
        threshold=threshold,
        constraints=constraints,
    )


async def classify_schema_async(
    text: str | Path,
    *,
    tasks: Mapping[str, Mapping[str, Any] | Sequence[str]],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.5,
    constraints: Sequence[Mapping[str, Any]] | None = None,
) -> SchemaClassification:
    return await asyncio.to_thread(
        classify_schema,
        text,
        tasks=tasks,
        backend=backend,
        device=device,
        model=model,
        threshold=threshold,
        constraints=constraints,
    )


def extract_graph(
    text: str | Path,
    *,
    entities: Sequence[str] | Mapping[str, str],
    relations: Sequence[Mapping[str, Any]],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    no_self_loops: bool = True,
    long: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> GraphExtraction:
    content = load_text_input(text)
    return get_extraction_backend(backend).extract_graph(
        content,
        entities,
        relations,
        device=device,
        model=model,
        no_self_loops=no_self_loops,
        long=long,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


async def extract_graph_async(
    text: str | Path,
    *,
    entities: Sequence[str] | Mapping[str, str],
    relations: Sequence[Mapping[str, Any]],
    backend: str = DEFAULT_BACKEND,
    device: str = DEFAULT_DEVICE,
    model: str = DEFAULT_MODEL,
    no_self_loops: bool = True,
    long: bool = False,
    chunk_size: int = 384,
    chunk_overlap: int = 64,
) -> GraphExtraction:
    return await asyncio.to_thread(
        extract_graph,
        text,
        entities=entities,
        relations=relations,
        backend=backend,
        device=device,
        model=model,
        no_self_loops=no_self_loops,
        long=long,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


class ExtractEntitiesTask(BaseTask):
    name = "extract-entities"

    def run(
        self,
        input: str | Path,
        *,
        labels: Sequence[str] | Mapping[str, str] | None = None,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        threshold: float | None = None,
        long: bool | None = None,
        **overrides: Any,
    ) -> EntityExtraction:
        options = self._resolve_options(
            labels=labels,
            backend=backend,
            device=device,
            threshold=threshold,
            long=long,
            **overrides,
        )
        checkpoint = model if isinstance(model, str) else options.pop("checkpoint", None)
        options.setdefault("backend", DEFAULT_BACKEND)
        if checkpoint is not None:
            options["model"] = checkpoint
        if "labels" not in options or options["labels"] is None:
            raise TypeError("extract-entities requires labels.")
        return extract_entities(input, **options)


class ExtractRecordsTask(BaseTask):
    name = "extract-records"

    def run(
        self,
        input: str | Path,
        *,
        schema: Mapping[str, Any] | None = None,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        threshold: float | None = None,
        long: bool | None = None,
        **overrides: Any,
    ) -> RecordExtraction:
        options = self._resolve_options(
            schema=schema,
            backend=backend,
            device=device,
            threshold=threshold,
            long=long,
            **overrides,
        )
        checkpoint = model if isinstance(model, str) else options.pop("checkpoint", None)
        options.setdefault("backend", DEFAULT_BACKEND)
        if checkpoint is not None:
            options["model"] = checkpoint
        if "schema" not in options or options["schema"] is None:
            raise TypeError("extract-records requires a schema mapping.")
        return extract_records(input, **options)


class ClassifySchemaTask(BaseTask):
    name = "classify-schema"

    def run(
        self,
        input: str | Path,
        *,
        tasks: Mapping[str, Mapping[str, Any] | Sequence[str]] | None = None,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        threshold: float | None = None,
        constraints: Sequence[Mapping[str, Any]] | None = None,
        **overrides: Any,
    ) -> SchemaClassification:
        options = self._resolve_options(
            tasks=tasks,
            backend=backend,
            device=device,
            threshold=threshold,
            constraints=constraints,
            **overrides,
        )
        checkpoint = model if isinstance(model, str) else options.pop("checkpoint", None)
        options.setdefault("backend", DEFAULT_BACKEND)
        if checkpoint is not None:
            options["model"] = checkpoint
        if "tasks" not in options or options["tasks"] is None:
            raise TypeError("classify-schema requires a tasks mapping.")
        return classify_schema(input, **options)


class ExtractGraphTask(BaseTask):
    name = "extract-graph"

    def run(
        self,
        input: str | Path,
        *,
        entities: Sequence[str] | Mapping[str, str] | None = None,
        relations: Sequence[Mapping[str, Any]] | None = None,
        backend: str | None = None,
        device: str | None = None,
        model: str | None = None,
        no_self_loops: bool | None = None,
        long: bool | None = None,
        **overrides: Any,
    ) -> GraphExtraction:
        options = self._resolve_options(
            entities=entities,
            relations=relations,
            backend=backend,
            device=device,
            no_self_loops=no_self_loops,
            long=long,
            **overrides,
        )
        checkpoint = model if isinstance(model, str) else options.pop("checkpoint", None)
        options.setdefault("backend", DEFAULT_BACKEND)
        if checkpoint is not None:
            options["model"] = checkpoint
        if "entities" not in options or options["entities"] is None:
            raise TypeError("extract-graph requires entities.")
        if "relations" not in options or options["relations"] is None:
            raise TypeError("extract-graph requires relations.")
        return extract_graph(input, **options)


TASK_SPEC = (
    TaskSpec(
        name=ExtractEntitiesTask.name,
        task_factory=ExtractEntitiesTask,
        aliases=("extract_entities",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_backend=True,
        accepts_labels=True,
        accepts_device=True,
        accepts_threshold=True,
        accepts_checkpoint=True,
        requires_labels=True,
    ),
    TaskSpec(
        name=ExtractRecordsTask.name,
        task_factory=ExtractRecordsTask,
        aliases=("extract_records",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_backend=True,
        accepts_device=True,
        accepts_threshold=True,
        accepts_checkpoint=True,
    ),
    TaskSpec(
        name=ClassifySchemaTask.name,
        task_factory=ClassifySchemaTask,
        aliases=("classify_schema",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_backend=True,
        accepts_device=True,
        accepts_threshold=True,
        accepts_checkpoint=True,
    ),
    TaskSpec(
        name=ExtractGraphTask.name,
        task_factory=ExtractGraphTask,
        aliases=("extract_graph",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_backend=True,
        accepts_device=True,
        accepts_checkpoint=True,
    ),
)

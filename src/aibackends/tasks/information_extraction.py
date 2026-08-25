from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aibackends.backends.information_extraction import (
    EntityTypes,
    get_information_extraction_backend,
)
from aibackends.tasks._utils import load_text_input

DEFAULT_INFORMATION_EXTRACTION_BACKEND = "gliner25"


def extract_entities(
    text: str | Path,
    entity_types: EntityTypes,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    **options: Any,
) -> dict[str, Any]:
    """Extract source-grounded entities with a dedicated encoder backend."""
    content = load_text_input(text)
    return get_information_extraction_backend(backend).extract_entities(
        content,
        entity_types,
        model=model,
        device=device,
        **options,
    )


async def extract_entities_async(
    text: str | Path,
    entity_types: EntityTypes,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    **options: Any,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        extract_entities,
        text,
        entity_types,
        backend=backend,
        model=model,
        device=device,
        **options,
    )


def extract_entities_long(
    text: str | Path,
    entity_types: EntityTypes,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    **options: Any,
) -> dict[str, Any]:
    """Extract entities from a complete document with globally remapped offsets."""
    content = load_text_input(text)
    return get_information_extraction_backend(backend).extract_entities_long(
        content,
        entity_types,
        model=model,
        device=device,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **options,
    )


async def extract_entities_long_async(
    text: str | Path,
    entity_types: EntityTypes,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    chunk_size: int = 384,
    chunk_overlap: int = 64,
    **options: Any,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        extract_entities_long,
        text,
        entity_types,
        backend=backend,
        model=model,
        device=device,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **options,
    )


def batch_extract_entities(
    texts: Sequence[str | Path],
    entity_types: EntityTypes,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    batch_size: int = 8,
    **options: Any,
) -> list[dict[str, Any]]:
    """Extract entities with the backend's native batch implementation."""
    contents = [load_text_input(text) for text in texts]
    return get_information_extraction_backend(backend).batch_extract_entities(
        contents,
        entity_types,
        model=model,
        device=device,
        batch_size=batch_size,
        **options,
    )


async def batch_extract_entities_async(
    texts: Sequence[str | Path],
    entity_types: EntityTypes,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    batch_size: int = 8,
    **options: Any,
) -> list[dict[str, Any]]:
    return await asyncio.to_thread(
        batch_extract_entities,
        texts,
        entity_types,
        backend=backend,
        model=model,
        device=device,
        batch_size=batch_size,
        **options,
    )


def extract_schema(
    text: str | Path,
    schema: Any,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    **options: Any,
) -> dict[str, Any]:
    """Run a combined information-extraction schema through a backend."""
    content = load_text_input(text)
    return get_information_extraction_backend(backend).extract_schema(
        content,
        schema,
        model=model,
        device=device,
        **options,
    )


async def extract_schema_async(
    text: str | Path,
    schema: Any,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    **options: Any,
) -> dict[str, Any]:
    return await asyncio.to_thread(
        extract_schema,
        text,
        schema,
        backend=backend,
        model=model,
        device=device,
        **options,
    )


def classify_schema(
    text: str | Path,
    schema: Any,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    config: Any = None,
) -> Any:
    """Classify text under cross-task constraints."""
    content = load_text_input(text)
    return get_information_extraction_backend(backend).classify_schema(
        content,
        schema,
        model=model,
        device=device,
        config=config,
    )


async def classify_schema_async(
    text: str | Path,
    schema: Any,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    config: Any = None,
) -> Any:
    return await asyncio.to_thread(
        classify_schema,
        text,
        schema,
        backend=backend,
        model=model,
        device=device,
        config=config,
    )


def extract_graph(
    text: str | Path,
    schema: Any,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    config: Any = None,
) -> Any:
    """Extract a typed, constrained entity-relation graph."""
    content = load_text_input(text)
    return get_information_extraction_backend(backend).extract_graph(
        content,
        schema,
        model=model,
        device=device,
        config=config,
    )


async def extract_graph_async(
    text: str | Path,
    schema: Any,
    *,
    backend: str = DEFAULT_INFORMATION_EXTRACTION_BACKEND,
    model: str | None = None,
    device: str = "auto",
    config: Any = None,
) -> Any:
    return await asyncio.to_thread(
        extract_graph,
        text,
        schema,
        backend=backend,
        model=model,
        device=device,
        config=config,
    )

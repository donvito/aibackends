from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from aibackends.schemas.extraction import (
    EntityExtraction,
    GraphExtraction,
    RecordExtraction,
    SchemaClassification,
)

ConstraintSpec = Mapping[str, Any]
RelationSpec = Mapping[str, Any]
TaskSpecDict = Mapping[str, Any]


class BaseExtractionBackend(ABC):
    name: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @abstractmethod
    def load(self, *, device: str = "cpu", model: str | None = None) -> Any:
        """Load and cache the extraction model on the selected device."""

    @abstractmethod
    def extract_entities(
        self,
        text: str,
        labels: Sequence[str] | Mapping[str, str],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = 0.5,
        long: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        attributes: Mapping[str, Any] | None = None,
    ) -> EntityExtraction:
        """Extract labeled entity spans, optionally over a long document."""

    @abstractmethod
    def extract_entities_batch(
        self,
        texts: Sequence[str],
        labels: Sequence[str] | Mapping[str, str],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = 0.5,
        long: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        attributes: Mapping[str, Any] | None = None,
        batch_size: int = 8,
    ) -> list[EntityExtraction]:
        """Extract labeled entity spans from multiple texts in one model batch."""

    @abstractmethod
    def extract_records(
        self,
        text: str,
        schema: Mapping[str, Any],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = 0.5,
        long: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> RecordExtraction:
        """Extract structured JSON records from text."""

    @abstractmethod
    def classify_schema(
        self,
        text: str,
        tasks: Mapping[str, TaskSpecDict | Sequence[str]],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = 0.5,
        constraints: Sequence[ConstraintSpec] | None = None,
    ) -> SchemaClassification:
        """Classify text, optionally under cross-task constraints."""

    @abstractmethod
    def extract_graph(
        self,
        text: str,
        entities: Sequence[str] | Mapping[str, str],
        relations: Sequence[RelationSpec],
        *,
        device: str = "cpu",
        model: str | None = None,
        no_self_loops: bool = True,
        long: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> GraphExtraction:
        """Extract a typed entity-relation graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from aibackends.schemas.extraction import (
    AttributeSpec,
    ClassificationConstraint,
    ClassificationTaskSpec,
    ConstrainedClassification,
    EntityExtraction,
    KnowledgeGraph,
    RelationExtraction,
    RelationSpec,
)


class BaseExtractionBackend(ABC):
    name: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @abstractmethod
    def resolve_model_id(self, model: str | None) -> str:
        """Return the Hugging Face model id for ``model`` (alias or full id)."""

    @abstractmethod
    def load(self, *, device: str = "cpu", model: str | None = None) -> Any:
        """Load and cache the extractor on the selected device."""

    @abstractmethod
    def extract_entities(
        self,
        text: str,
        labels: list[str] | dict[str, str],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = 0.5,
        long_document: bool | None = None,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> EntityExtraction:
        """Extract labeled spans from ``text``."""

    @abstractmethod
    def extract_relations(
        self,
        text: str,
        relations: list[str],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = 0.5,
    ) -> RelationExtraction:
        """Extract independently decoded relation triples."""

    @abstractmethod
    def extract_graph(
        self,
        text: str,
        entities: list[str],
        relations: list[RelationSpec],
        *,
        device: str = "cpu",
        model: str | None = None,
        no_self_loops: bool = True,
        long_document: bool | None = None,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        beam_size: int = 32,
    ) -> KnowledgeGraph:
        """Extract a schema-consistent entity-relation graph."""

    @abstractmethod
    def classify_constrained(
        self,
        text: str,
        tasks: list[ClassificationTaskSpec],
        constraints: list[ClassificationConstraint],
        *,
        device: str = "cpu",
        model: str | None = None,
        long_document: bool | None = None,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        decoder: str = "exact",
        beam_size: int = 16,
    ) -> ConstrainedClassification:
        """Classify ``text`` under declared cross-task constraints."""

    @abstractmethod
    def extract_with_attributes(
        self,
        text: str,
        labels: list[str] | dict[str, str],
        attributes: list[AttributeSpec],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = 0.5,
        long_document: bool | None = None,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> EntityExtraction:
        """Extract spans and decode per-span attributes in one pass."""

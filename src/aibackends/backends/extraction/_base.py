from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from aibackends.schemas.extraction import (
    EntityExtraction,
    KnowledgeGraph,
    TextClassification,
)

LabelsInput = Sequence[str] | Mapping[str, str]
AttributesInput = Mapping[str, Mapping[str, Any]]
TasksInput = Mapping[str, Sequence[str] | Mapping[str, Any]]
ConstraintsInput = Sequence[Mapping[str, Any]]
RelationsInput = Sequence[Mapping[str, Any]]


class BaseExtractionBackend(ABC):
    name: str
    aliases: tuple[str, ...] = ()
    default_model: str

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @abstractmethod
    def resolve_model_id(self, model: str | None) -> str:
        """Map a model variant or repo id to the concrete model id."""

    @abstractmethod
    def load(self, *, model: str | None = None, device: str = "cpu") -> Any:
        """Load and cache the extraction model on the selected device."""

    @abstractmethod
    def extract_entities(
        self,
        text: str,
        *,
        labels: LabelsInput,
        attributes: AttributesInput | None = None,
        model: str | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        long_document: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> EntityExtraction:
        """Extract typed spans (optionally with span attributes) from one text."""

    @abstractmethod
    def extract_entities_batch(
        self,
        texts: Sequence[str],
        *,
        labels: LabelsInput,
        attributes: AttributesInput | None = None,
        model: str | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        batch_size: int = 8,
        long_document: bool = False,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
    ) -> list[EntityExtraction]:
        """Extract typed spans from multiple texts in one model batch."""

    @abstractmethod
    def classify_text(
        self,
        text: str,
        *,
        tasks: TasksInput,
        constraints: ConstraintsInput | None = None,
        model: str | None = None,
        device: str = "cpu",
    ) -> TextClassification:
        """Classify one text across one or more tasks with optional constraints."""

    @abstractmethod
    def classify_text_batch(
        self,
        texts: Sequence[str],
        *,
        tasks: TasksInput,
        constraints: ConstraintsInput | None = None,
        model: str | None = None,
        device: str = "cpu",
        batch_size: int = 8,
    ) -> list[TextClassification]:
        """Classify multiple texts in one model batch."""

    @abstractmethod
    def extract_graph(
        self,
        text: str,
        *,
        entities: LabelsInput,
        relations: RelationsInput,
        no_self_loops: bool = True,
        model: str | None = None,
        device: str = "cpu",
        optimizer: str = "beam",
        beam_size: int = 32,
    ) -> KnowledgeGraph:
        """Jointly extract a consistent entity-relation graph from one text."""

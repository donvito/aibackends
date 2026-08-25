from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

EntityTypes = Sequence[str] | Mapping[str, Any]


class BaseInformationExtractionBackend(ABC):
    """Backend contract for schema-driven encoder information extraction."""

    name: str
    aliases: tuple[str, ...] = ()
    model_ids: Mapping[str, str]
    default_model: str

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @abstractmethod
    def resolve_model_id(self, model: str | None = None) -> str:
        """Resolve a backend model alias to its full model ID."""

    @abstractmethod
    def normalize_device(self, device: str) -> str:
        """Normalize or auto-detect an inference device."""

    @abstractmethod
    def load(self, *, model: str | None = None, device: str = "auto") -> Any:
        """Load and cache a model for one checkpoint and device."""

    @abstractmethod
    def create_schema(self, *, model: str | None = None, device: str = "auto") -> Any:
        """Create a combined extraction schema builder."""

    @abstractmethod
    def create_attribute_group(
        self,
        labels: Sequence[str],
        **options: Any,
    ) -> Any:
        """Create a span-attribute group without exposing the native dependency."""

    @abstractmethod
    def create_classification_schema(self) -> Any:
        """Create a constrained-classification schema builder."""

    @abstractmethod
    def create_classification_config(self, **options: Any) -> Any:
        """Create call-scoped constrained-classification settings."""

    @property
    @abstractmethod
    def classification_constraints(self) -> Any:
        """Return the constrained-classification expression helpers."""

    @abstractmethod
    def create_joint_schema(
        self,
        *,
        model: str | None = None,
        device: str = "auto",
    ) -> Any:
        """Create a typed Joint IE graph schema builder."""

    @abstractmethod
    def create_joint_config(self, **options: Any) -> Any:
        """Create call-scoped Joint IE decoding settings."""

    @abstractmethod
    def extract_entities(
        self,
        text: str,
        entity_types: EntityTypes,
        *,
        model: str | None = None,
        device: str = "auto",
        **options: Any,
    ) -> dict[str, Any]:
        """Extract entities from one text."""

    @abstractmethod
    def extract_entities_long(
        self,
        text: str,
        entity_types: EntityTypes,
        *,
        model: str | None = None,
        device: str = "auto",
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        **options: Any,
    ) -> dict[str, Any]:
        """Extract entities across overlapping full-document chunks."""

    @abstractmethod
    def batch_extract_entities(
        self,
        texts: Sequence[str],
        entity_types: EntityTypes,
        *,
        model: str | None = None,
        device: str = "auto",
        batch_size: int = 8,
        **options: Any,
    ) -> list[dict[str, Any]]:
        """Extract entities from a native model batch."""

    @abstractmethod
    def extract_schema(
        self,
        text: str,
        schema: Any,
        *,
        model: str | None = None,
        device: str = "auto",
        **options: Any,
    ) -> dict[str, Any]:
        """Run a combined entity, classification, relation, or record schema."""

    @abstractmethod
    def classify_schema(
        self,
        text: str,
        schema: Any,
        *,
        model: str | None = None,
        device: str = "auto",
        config: Any = None,
    ) -> Any:
        """Run constrained classification and preserve its typed result."""

    @abstractmethod
    def extract_graph(
        self,
        text: str,
        schema: Any,
        *,
        model: str | None = None,
        device: str = "auto",
        config: Any = None,
    ) -> Any:
        """Run Joint IE and preserve its typed graph result."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from aibackends.schemas.moderation import PromptModeration, ResponseModeration


class BaseModerationBackend(ABC):
    name: str
    model_id: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @abstractmethod
    def load(self, *, device: str = "cpu") -> Any:
        """Load and cache the moderation model on the selected device."""

    @abstractmethod
    def moderate_prompt(
        self,
        prompt: str,
        *,
        device: str = "cpu",
        threshold: float = 0.5,
        category_threshold: float = 0.4,
    ) -> PromptModeration:
        """Classify one user prompt for safety, toxicity, and jailbreak strategies."""

    @abstractmethod
    def moderate_response(
        self,
        response: str,
        *,
        prompt: str | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        category_threshold: float = 0.4,
    ) -> ResponseModeration:
        """Classify one model response for safety, toxicity, and refusal."""

    @abstractmethod
    def moderate_prompts(
        self,
        prompts: Sequence[str],
        *,
        device: str = "cpu",
        threshold: float = 0.5,
        category_threshold: float = 0.4,
        batch_size: int = 8,
    ) -> list[PromptModeration]:
        """Classify multiple prompts in one model batch."""

    @abstractmethod
    def moderate_responses(
        self,
        responses: Sequence[str],
        *,
        prompts: Sequence[str | None] | None = None,
        device: str = "cpu",
        threshold: float = 0.5,
        category_threshold: float = 0.4,
        batch_size: int = 8,
    ) -> list[ResponseModeration]:
        """Classify multiple responses in one model batch."""

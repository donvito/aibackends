from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from aibackends.schemas.routing import RoutingResult


class BaseRoutingBackend(ABC):
    name: str
    model_id: str
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    @abstractmethod
    def load(self, *, device: str = "cpu") -> Any:
        """Load and cache the routing model on the selected device."""

    @abstractmethod
    def route(
        self,
        text: str,
        routes: Sequence[str],
        *,
        device: str = "cpu",
        threshold: float | None = None,
    ) -> RoutingResult:
        """Score one text against every route and return the ranked result."""

    @abstractmethod
    def route_batch(
        self,
        texts: Sequence[str],
        routes: Sequence[str],
        *,
        device: str = "cpu",
        threshold: float | None = None,
    ) -> list[RoutingResult]:
        """Route multiple texts against the same set of routes."""

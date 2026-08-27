from __future__ import annotations

from aibackends.core.types import AIBackendsModel


class RouteScore(AIBackendsModel):
    route: str
    score: float


class RoutingResult(AIBackendsModel):
    """Ranked zero-shot routing decision for one text.

    ``scores`` is sorted by descending score and only contains routes at or
    above the requested threshold, so ``best_route`` is ``None`` when every
    route was filtered out.
    """

    text: str
    best_route: str | None
    scores: list[RouteScore]
    backend_used: str
    model_id: str

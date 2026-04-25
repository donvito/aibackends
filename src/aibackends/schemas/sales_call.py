from __future__ import annotations

from aibackends.core.types import AIBackendsModel


class SalesCallReport(AIBackendsModel):
    talk_ratio: dict[str, float]
    objections: list[str]
    buying_signals: list[str]
    action_items: list[str]
    score: float
    sentiment: str | None = None

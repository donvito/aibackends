from __future__ import annotations

from aibackends.core.types import AIBackendsModel


class LineItem(AIBackendsModel):
    description: str
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None

from __future__ import annotations

from aibackends.core.types import AIBackendsModel
from aibackends.schemas.common import LineItem


class InvoiceOutput(AIBackendsModel):
    vendor: str
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float
    due_date: str | None = None
    payment_terms: str | None = None

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.schemas.invoice import InvoiceOutput
from aibackends.tasks._utils import build_messages, load_text_input, run_structured_task


def extract_invoice(
    file_path: str | Path,
    *,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> InvoiceOutput:
    content = load_text_input(file_path)
    messages = build_messages(
        "You extract structured fields from invoices and receipts.",
        (
            "Extract the invoice vendor, line items, subtotal, tax, total, "
            "due date, and payment terms. "
            "If a field is missing, return null for optional values.\n\n"
            f"Invoice content:\n{content}"
        ),
    )
    return run_structured_task(
        task_name="extract_invoice",
        schema=InvoiceOutput,
        messages=messages,
        runtime=runtime,
        model=model,
        **overrides,
    )


async def extract_invoice_async(
    file_path: str | Path,
    *,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> InvoiceOutput:
    return await asyncio.to_thread(
        extract_invoice,
        file_path,
        runtime=runtime,
        model=model,
        **overrides,
    )

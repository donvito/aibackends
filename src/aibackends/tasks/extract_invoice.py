from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.core.registry import TaskSpec
from aibackends.schemas.invoice import InvoiceOutput
from aibackends.tasks._base import BaseTask
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


class ExtractInvoiceTask(BaseTask):
    name = "extract-invoice"

    def run(
        self,
        input: str | Path,
        *,
        runtime: str | None = None,
        model: str | None = None,
        **overrides: Any,
    ) -> InvoiceOutput:
        options = self._resolve_options(runtime=runtime, model=model, **overrides)
        return extract_invoice(input, **options)


TASK_SPEC = TaskSpec(
    name=ExtractInvoiceTask.name,
    task_factory=ExtractInvoiceTask,
    aliases=("extract_invoice",),
)

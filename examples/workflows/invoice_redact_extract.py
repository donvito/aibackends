"""Invoice PDF redact + extract pipeline.

Reads ``examples/data/pdf/invoice.pdf``, extracts text from the PDF, redacts
likely PII with GLiNER, then extracts invoice fields into structured JSON.

This is text-first rather than vision-first, so it works best when the PDF
already contains extractable text. The sample invoice is mostly business data,
so GLiNER may redact only a few fields or none at all.

Requires:
    pip install 'aibackends[pdf]'
    pip install 'aibackends[pii]'
    pip install 'aibackends[llamacpp]'
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.schemas.invoice import InvoiceOutput
from aibackends.steps.enrich import LLMAnalyser
from aibackends.steps.enrich import PIIRedactor as PIIRedactorStep
from aibackends.steps.ingest import PDFIngestor
from aibackends.workflows import Pipeline


class PIISummary(BaseModel):
    backend: str
    entities_redacted: int
    redaction_map: dict[str, str]


class InvoiceExtractionResult(BaseModel):
    source: str
    redacted_text: str
    invoice: InvoiceOutput
    pii: PIISummary


# Keep labels focused on personal/contact data so invoice totals and vendor-style
# business fields remain available for extraction.
INVOICE_PII_LABELS = [
    "person_name",
    "email",
    "phone_number",
    "address",
    "account_number",
    "tax_identifier",
]

INVOICE_SYSTEM_PROMPT = "You extract structured invoice data from redacted invoice text."

INVOICE_USER_PROMPT = (
    "Extract the invoice vendor, line items, subtotal, tax, total, due date, "
    "and payment terms from the redacted invoice text below.\n"
    "Keep bracketed placeholders such as [PERSON_NAME_1] exactly as written.\n"
    "Do not guess redacted values.\n"
    "If a field is missing, return null for optional values."
)


class InvoiceRedactExtractWorkflow(Pipeline):
    steps = [
        PDFIngestor(),
        PIIRedactorStep(backend="gliner", labels=INVOICE_PII_LABELS),
        LLMAnalyser(
            schema=InvoiceOutput,
            task_name="invoice-redacted-extract",
            system_prompt=INVOICE_SYSTEM_PROMPT,
            prompt=INVOICE_USER_PROMPT,
            input_key="text",
            output_key="invoice",
        ),
    ]


def main() -> None:
    try:
        invoice_path = Path(__file__).parent.parent / "data" / "pdf" / "invoice.pdf"
        workflow = InvoiceRedactExtractWorkflow(runtime=LLAMACPP, model=GEMMA4_E2B)
        raw_result = workflow.run(invoice_path)
        redaction = raw_result["pii_redaction"]
        result = InvoiceExtractionResult(
            source=str(raw_result.get("path", "")),
            redacted_text=raw_result.get("text", ""),
            invoice=raw_result["invoice"],
            pii=PIISummary(
                backend=redaction.backend_used,
                entities_redacted=len(redaction.entities_found),
                redaction_map=redaction.redaction_map,
            ),
        )
        print(result.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

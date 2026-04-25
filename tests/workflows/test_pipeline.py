from __future__ import annotations

import aibackends.steps.enrich as enrich_module
from aibackends.schemas.invoice import InvoiceOutput
from aibackends.schemas.pii import RedactedText
from aibackends.steps.enrich import LLMAnalyser
from aibackends.steps.ingest import FileIngestor
from aibackends.steps.validate import PydanticValidator
from aibackends.workflows import InvoiceProcessor, SalesCallAnalyser
from aibackends.workflows._base import Pipeline


def test_invoice_processor_batch_collects_results(tmp_path):
    file_a = tmp_path / "invoice-a.txt"
    file_b = tmp_path / "invoice-b.txt"
    file_a.write_text("Invoice from Acme Corp for consulting services")
    file_b.write_text("Invoice from Acme Corp for support services")

    result = InvoiceProcessor(runtime="stub", model="stub-model").run_batch(
        inputs=[file_a, file_b],
        max_concurrency=2,
        on_error="collect",
    )

    assert len(result.results) == 2
    assert result.errors == []
    assert all(item.total == 1250.0 for item in result.results)


def test_sales_call_analyser_accepts_transcript_file(tmp_path, monkeypatch):
    transcript = tmp_path / "call.txt"
    transcript.write_text("Agent: Thanks for joining.\nCustomer: Price is a concern.")

    monkeypatch.setattr(
        enrich_module,
        "redact_pii",
        lambda text, backend="gliner", labels=None: RedactedText(
            original_text=str(text),
            redacted_text=str(text),
            entities_found=[],
            redaction_map={},
            backend_used=backend,
        ),
    )

    result = SalesCallAnalyser(runtime="stub", model="stub-model").run(transcript)
    assert result.score == 7.4
    assert result.sentiment == "positive"


def test_custom_pipeline_passes_runtime_config_to_llm_step(tmp_path):
    class CustomInvoicePipeline(Pipeline):
        steps = [
            FileIngestor(),
            LLMAnalyser(schema=InvoiceOutput, prompt="Extract invoice data."),
            PydanticValidator(schema=InvoiceOutput),
        ]

    invoice = tmp_path / "invoice.txt"
    invoice.write_text("Invoice from Acme Corp for consulting services")

    result = CustomInvoicePipeline(runtime="stub", model="stub-model").run(invoice)

    assert result.total == 1250.0

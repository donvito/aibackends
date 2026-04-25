from __future__ import annotations

import aibackends.steps.enrich as enrich_module
from aibackends.schemas.invoice import InvoiceOutput
from aibackends.schemas.pii import RedactedText
from aibackends.steps.enrich import LLMAnalyser, LLMTextGenerator, TaskRunner
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


def test_llm_analyser_can_store_result_under_output_key(tmp_path):
    class CustomInvoicePipeline(Pipeline):
        steps = [
            FileIngestor(),
            LLMAnalyser(
                schema=InvoiceOutput,
                prompt="Extract invoice data.",
                task_name="invoice-extract",
                input_key="text",
                output_key="invoice",
            ),
        ]

    invoice = tmp_path / "invoice.txt"
    invoice.write_text("Invoice from Acme Corp for consulting services")

    result = CustomInvoicePipeline(runtime="stub", model="stub-model").run(invoice)

    assert result["text"] == "Invoice from Acme Corp for consulting services"
    assert result["invoice"].total == 1250.0


def test_llm_text_generator_can_store_text_under_output_key(tmp_path):
    class SummaryPipeline(Pipeline):
        steps = [
            FileIngestor(),
            LLMTextGenerator(
                prompt="Summarize this document.",
                task_name="summarize",
                input_key="text",
                output_key="summary",
            ),
        ]

    note = tmp_path / "note.txt"
    note.write_text("Alice owns pricing and Bob owns analytics.")

    result = SummaryPipeline(runtime="stub", model="stub-model").run(note)

    assert result["text"] == "Alice owns pricing and Bob owns analytics."
    assert result["summary"] == "Stub summary"


def test_task_runner_executes_registered_task_with_output_key(tmp_path):
    class ClassificationPipeline(Pipeline):
        steps = [
            FileIngestor(),
            TaskRunner(
                task_name="classify",
                input_key="text",
                output_key="classification",
                task_config={"labels": ["invoice", "contract", "receipt"]},
            ),
        ]

    document = tmp_path / "doc.txt"
    document.write_text("Invoice from Acme Corp for consulting services")

    result = ClassificationPipeline(runtime="stub", model="stub-model").run(document)

    assert result["text"] == "Invoice from Acme Corp for consulting services"
    assert result["classification"].label == "invoice"

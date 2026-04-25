from aibackends.core.registry import WorkflowSpec
from aibackends.schemas.invoice import InvoiceOutput
from aibackends.steps.enrich import VisionExtractor
from aibackends.steps.ingest import PDFIngestor
from aibackends.steps.process import ImageRenderer
from aibackends.steps.validate import PydanticValidator
from aibackends.workflows._base import Pipeline


class InvoiceProcessor(Pipeline):
    steps = [
        PDFIngestor(),
        ImageRenderer(dpi=150),
        VisionExtractor(schema=InvoiceOutput, prompt="Extract invoice data from the document."),
        PydanticValidator(schema=InvoiceOutput),
    ]


WORKFLOW_SPEC = WorkflowSpec(
    name="invoice",
    workflow_factory=InvoiceProcessor,
    aliases=("invoice-processor", "invoice_processor"),
)

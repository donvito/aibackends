# Custom Pipelines

Subclass `Pipeline` and compose steps.

```python
from aibackends.schemas.invoice import InvoiceOutput
from aibackends.steps.enrich import VisionExtractor
from aibackends.steps.ingest import PDFIngestor
from aibackends.steps.process import ImageRenderer
from aibackends.steps.validate import PydanticValidator
from aibackends.workflows import Pipeline


class ContractReviewer(Pipeline):
    steps = [
        PDFIngestor(),
        ImageRenderer(dpi=150),
        VisionExtractor(schema=InvoiceOutput, prompt="Extract the contract terms."),
        PydanticValidator(schema=InvoiceOutput),
    ]
```

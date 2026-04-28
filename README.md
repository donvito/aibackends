# AIBackends

Run AI tasks and workflows locally.

Build extraction, classification, embeddings, redaction, and analysis pipelines
in plain Python with `llamacpp` and `transformers`.

- First-class `llamacpp` and `transformers` runtimes
- Typed outputs for extraction and analysis tasks
- Reusable tasks and workflows for scripts, apps, and batch jobs
- Practical local examples for text, image OCR, documents, audio, and video

## Install

```bash
pip install aibackends

# Local runtimes
pip install aibackends[llamacpp]
pip install aibackends[llamacpp-cuda]
pip install aibackends[llamacpp-metal]
pip install aibackends[transformers]

# Capability extras
pip install aibackends[pdf]
pip install aibackends[audio]
pip install aibackends[video]
pip install aibackends[pii]
```

## Quickstart

**Extract an invoice locally**

```python
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import ExtractInvoiceTask, create_task

task = create_task(
    ExtractInvoiceTask,
    runtime=LLAMACPP,
    model=GEMMA4_E2B,
)

result = task.run("invoice.pdf")
print(result.total)
```

## Examples

### Single tasks

**Classify text locally and redact PII**

```python
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import ClassifyTask, RedactPIITask, create_task

classifier = create_task(
    ClassifyTask,
    runtime=LLAMACPP,
    model=GEMMA4_E2B,
    labels=["invoice", "contract", "receipt"],
)
redactor = create_task(
    RedactPIITask,
    backend="gliner",
    labels=["email", "phone_number"],
)

classification = classifier.run("invoice text")
redacted = redactor.run("john@example.com called from +1 555 0100")
```
`RedactPIITask` uses a dedicated backend such as `gliner` or `openai-privacy`
(the local `privacy-filter` model) rather than the general LLM runtime
interface.

**Generate local embeddings**

```python
from aibackends.models import MINILM_L6
from aibackends.runtimes import TRANSFORMERS
from aibackends.tasks import EmbedTask, create_task

embedder = create_task(
    EmbedTask,
    runtime=TRANSFORMERS,
    model=MINILM_L6,
)

vector = embedder.run("Payments failed after checkout deploy.")
print(len(vector))
print(vector[:5])
```

### Workflows

**Batch-process sales calls analysis locally**

```python
from pathlib import Path

from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.workflows import SalesCallAnalyser, create_workflow

workflow = create_workflow(
    SalesCallAnalyser,
    runtime=LLAMACPP,
    model=GEMMA4_E2B,
)

results = workflow.run_batch(
    inputs=Path("./calls").glob("*.m4a"),
    max_concurrency=4,
    on_error="collect",
)
```

**Run OCR locally**

```python
from pydantic import BaseModel, Field

from aibackends.models import QWEN3_VL_4B
from aibackends.runtimes import LLAMACPP
from aibackends.schemas.common import LineItem
from aibackends.steps.enrich import VisionExtractor
from aibackends.steps.ingest import ImageIngestor
from aibackends.workflows import Pipeline


class Receipt(BaseModel):
    merchant: str | None = None
    total: float | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class ReceiptOCR(Pipeline):
    steps = [
        ImageIngestor(),
        VisionExtractor(
            schema=Receipt,
            prompt="Extract merchant, total, and line_items from this receipt.",
        ),
    ]


result = ReceiptOCR(runtime=LLAMACPP, model=QWEN3_VL_4B).run("receipt.jpeg")
print(result.model_dump_json(indent=2))
```

## Included

- Local runtimes: `llamacpp`, `transformers`
- Tasks: `summarize`, `extract`, `classify`, `embed`, `extract_invoice`,
  `redact_pii`, `analyse_sales_call`, `analyse_video_ad`
- Workflows: `InvoiceProcessor`, `PIIRedactor`, `SalesCallAnalyser`,
  `VideoAdIntelligence`
- Outputs: `InvoiceOutput`, `SalesCallReport`, `VideoAdReport`,
  `RedactedText`, `Classification`

Tool and agent integrations can be added later without changing the core task
and workflow layer.

## CLI

```bash
# Install the runtime or backend extra first
pip install 'aibackends[llamacpp]'
pip install 'aibackends[pii]'

aibackends task extract-invoice --input invoice.pdf --runtime llamacpp --model gemma4-e2b
aibackends task classify --input doc.txt --labels invoice,contract,receipt --runtime llamacpp --model gemma4-e2b
aibackends task redact-pii --input transcript.txt --backend gliner --labels email,phone_number
aibackends pull gemma4-e2b --runtime llamacpp
aibackends check llamacpp --model gemma4-e2b
```

Full command reference: `docs/cli.md`.

## Docs and Examples

- `docs/usage.md` for install, local runtimes, tasks, and workflows
- `docs/concepts.md` for task, runtime, backend, model, and workflow terms
- `docs/extending.md` for custom runtimes, backends, tasks, and workflows
- `docs/api-reference/index.md` for the public API
- `examples/README.md` for runnable examples, including local image OCR

## Development

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest tests
python3 -m mypy src tests
ruff check .
```

See `CONTRIBUTING.md` for contribution guidelines.
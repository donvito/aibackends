# AIBackends

Framework-agnostic AI tasks and workflows with first-class local runtimes for
Llama.cpp and Transformers.

Use the same typed AI tasks and workflows in plain Python or as tools in
LangGraph, pydantic-ai, OpenAI Agents SDK, CrewAI, Agno, and LlamaIndex.

- One API across local and hosted runtimes
- Built-in local support for `llamacpp` and `transformers`
- Structured Pydantic outputs for extraction and analysis tasks
- Optional workflows for batch jobs and pipelines

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

Extract an invoice:

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

`redact_pii` uses a dedicated backend such as `gliner` or `openai-privacy`
rather than the general LLM runtime interface.

## Examples

### Single tasks

Classify text and redact PII:

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

Generate embeddings:

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

Batch-process sales calls:

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

## Included

- Tasks: `summarize`, `extract`, `classify`, `embed`, `extract_invoice`,
  `redact_pii`, `analyse_sales_call`, `analyse_video_ad`
- Workflows: `InvoiceProcessor`, `PIIRedactor`, `SalesCallAnalyser`,
  `VideoAdIntelligence`
- Outputs: `InvoiceOutput`, `SalesCallReport`, `VideoAdReport`,
  `RedactedText`, `Classification`
- Runtimes: `llamacpp`, `transformers`, `ollama`, `lmstudio`, `anthropic`,
  `together`, `groq`

## CLI

```bash
aibackends task extract-invoice --input invoice.pdf
aibackends task redact-pii --input transcript.txt --backend gliner --labels email,phone_number
aibackends task classify --input doc.txt --labels invoice,contract,receipt
aibackends pull gemma4-e2b --runtime llamacpp
aibackends check transformers
```

Full command reference: `docs/cli.md`.

## Docs and Examples

- `docs/usage.md` for install, configuration, tasks, workflows, and integrations
- `docs/concepts.md` for task, runtime, backend, model, and workflow terms
- `docs/extending.md` for custom runtimes, backends, tasks, and workflows
- `docs/api-reference/index.md` for the public API
- `examples/README.md` for runnable examples

## Development

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest tests
python3 -m mypy src tests
ruff check .
```

See `CONTRIBUTING.md` for contribution guidelines.

# Usage

AIBackends is designed first for local model workflows. Most projects start by
configuring `llamacpp` or `transformers`, then reuse the same typed tasks and
workflows across scripts, apps, and batch jobs.

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

Downloaded local models for `llamacpp` and `aibackends pull` use the standard
Hugging Face cache by default, usually `~/.cache/huggingface/hub`.

## Configure Local Runtimes

Use `configure()` to set global local runtime defaults:

```python
from aibackends import configure
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP

configure(runtime=LLAMACPP, model=GEMMA4_E2B)
```

Use `available_runtimes()` and `available_models()` when you want to inspect the
curated Python-facing catalog of supported runtime and model refs.

In Python, pass typed refs such as `LLAMACPP` and `GEMMA4_E2B`. String names are
kept for text boundaries like CLI flags and YAML config files.

You can also load YAML:

```python
from aibackends import load_config

load_config("aibackends.yml")
```

For local `transformers` models, prompt rendering is configurable:

```python
from aibackends.models import GEMMA3_270M_IT
from aibackends.runtimes import TRANSFORMERS

configure(
    runtime=TRANSFORMERS,
    model=GEMMA3_270M_IT,
    prompt_format="auto",  # auto | chat_template | text
    # chat_template="...",
    # chat_template_path="template.jinja",
)
```

`prompt_format="auto"` prefers a configured template override, then the
tokenizer's own chat template, then plain text.

### Device and quantization

Both local runtimes accept a `device` toggle. `llamacpp` maps it to GPU layer
offload and `transformers` maps it to `device_map`:

```python
from aibackends import configure
from aibackends.models import LFM25_2_6B
from aibackends.runtimes import LLAMACPP

configure(
    runtime=LLAMACPP,
    model=LFM25_2_6B,
    device="cpu",  # "cpu" | "gpu" | None for auto-detect
)
```

For GGUF models on `llamacpp`, the quantization is configurable. Model
profiles can define their own default (for example `LFM25_2_6B` defaults to
`Q4_K_M`), and machines with more capacity can pick a larger file:

```python
configure(
    runtime=LLAMACPP,
    model=LFM25_2_6B,
    quantization="Q8_0",  # any quant published in the GGUF repo
)
```

When neither the config nor the model profile sets a quantization, the
hardware default is used (`Q5_K_M` with CUDA/Metal, `Q4_K_M` on CPU).

### LiquidAI LFM2.5-2.6B

`LFM25_2_6B` targets `LiquidAI/LFM2.5-2.6B` on `transformers` and
`LiquidAI/LFM2.5-2.6B-GGUF` on `llamacpp`. The profile applies the
generation defaults recommended by Liquid AI (`temperature=0.1`, `top_k=50`,
`repetition_penalty=1.1`, and `bfloat16` on `transformers`).

LFM2.5 is a reasoning model with native tool calling. See
`examples/tasks/tool_calling_lfm.py` for a runnable tool-calling demo on
either runtime.

### LiquidAI LFM2.5-VL-3B (vision)

`LFM25_VL_3B` targets `LiquidAI/LFM2.5-VL-3B-GGUF` on `llamacpp` (image
inputs are llama.cpp-only). The profile defaults to the `Q4_K_M` quantization
and applies the same generation defaults as LFM2.5-2.6B. The matching
`mmproj` projector is downloaded from the same repository automatically.

```python
from pydantic import BaseModel

from aibackends.models import LFM25_VL_3B
from aibackends.runtimes import LLAMACPP
from aibackends.steps.enrich import VisionExtractor
from aibackends.steps.ingest import ImageIngestor
from aibackends.workflows import Pipeline


class Receipt(BaseModel):
    merchant: str | None = None
    total: float | None = None


class ReceiptOCR(Pipeline):
    steps = [
        ImageIngestor(),
        VisionExtractor(
            schema=Receipt,
            prompt="Extract the merchant and total from this receipt.",
        ),
    ]


result = ReceiptOCR(runtime=LLAMACPP, model=LFM25_VL_3B, device="cpu").run(
    "receipt.jpeg"
)
```

The model is small enough for CPU inference with the default `Q4_K_M` GGUF;
see the committed CPU report in `benchmarks/reports/` (task benchmark,
`llamacpp`, device `cpu`) for latency numbers. A runnable receipt-extraction
demo lives at `examples/workflows/image_ocr_lfm.py`.

If you need a different runtime for one call, override it explicitly:

```python
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import TRANSFORMERS
from aibackends.tasks import extract_invoice

result = extract_invoice("invoice.pdf", runtime=TRANSFORMERS, model=GEMMA4_E2B)
```

## Call Tasks

```python
from aibackends.tasks import classify, redact_pii, summarize

summary = summarize("notes.txt")
classification = classify("invoice text", labels=["invoice", "contract", "receipt"])
redacted = redact_pii(
    "john@example.com called from +1 555 0100",
    backend="gliner",
    labels=["email", "phone_number"],
)
```

`redact_pii` uses a PII backend, not the configured runtime. Use `backend="gliner"`
or `backend="openai-privacy"` for the local `privacy-filter` model.

Every task also exposes an async variant with the `_async` suffix.

Tasks are also available as configured `BaseTask` objects through the factory:

```python
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import SummarizeTask, create_task

task = create_task(
    SummarizeTask,
    runtime=LLAMACPP,
    model=GEMMA4_E2B,
)
summary = task.run("notes.txt")
```

## Use Workflows

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

Included workflows:

- `InvoiceProcessor`
- `SalesCallAnalyser`
- `VideoAdIntelligence`
- `PIIRedactor`

Batch `on_error` supports `"raise"`, `"skip"`, and `"collect"`.

## CLI

```bash
aibackends task extract-invoice --input invoice.pdf
aibackends task redact-pii --input transcript.txt --backend gliner --labels email,phone_number,user_name
aibackends task classify --input doc.txt --labels invoice,contract,receipt
aibackends pull gemma4-e2b --runtime llamacpp
aibackends check transformers
```

For full command reference (subcommands, flags, output formats, and what is
not exposed via CLI), see the [CLI guide](cli.md).

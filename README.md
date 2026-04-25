# AIBackends

Pluggable AI tasks for any agent framework, powered by local models.

AIBackends is a Python library of ready-made AI tasks that plug into agent frameworks as tools. Extract invoices, redact PII, classify documents, analyse sales calls, and analyse video ads with one function call and a typed result.

```python
from aibackends import configure
from aibackends.tasks import extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")

result = extract_invoice("invoice.pdf")
print(result.total)
```

## Why AIBackends

- Framework agnostic: LangGraph, pydantic-ai, OpenAI Agents SDK, CrewAI, Agno, LlamaIndex, or your own code
- Local-first: first-class support for `llama-cpp-python` and `transformers`
- Typed outputs: every structured task returns a Pydantic model
- Optional orchestration: workflows add retries, steps, and batch execution without forcing a framework

## Install

```bash
pip install aibackends

# Local runtimes
pip install aibackends[llamacpp]
pip install aibackends[llamacpp-cuda]
pip install aibackends[llamacpp-metal]
pip install aibackends[transformers]

# Task extras
pip install aibackends[pdf]
pip install aibackends[audio]
pip install aibackends[video]
pip install aibackends[pii]
```

Downloaded local models for `llamacpp` and `aibackends pull` go into the standard Hugging Face cache by default, usually `~/.cache/huggingface/hub`.

If you want to inspect or clean up downloaded models:

```bash
cd ~/.cache/huggingface/hub
```

## Quickstart

```python
from aibackends import configure
from aibackends.tasks import classify, redact_pii, summarize

configure(runtime="llamacpp", model="gemma4-e2b")

summary = summarize("notes.txt")
classification = classify("invoice text", labels=["invoice", "contract", "receipt"])
redacted = redact_pii(
    "john@example.com called from +1 555 0100",
    backend="gliner",
    labels=["email", "phone_number"],
)
```

`redact_pii` uses its own PII detection backend such as `gliner` or `openai-privacy`; it does not use the `configure()` runtime/model settings. When using `gliner`, you can pass custom entity labels with `labels=[...]`.

## Tasks

```python
from aibackends.tasks import (
    analyse_sales_call,
    analyse_video_ad,
    classify,
    embed,
    extract,
    extract_invoice,
    redact_pii,
    summarize,
)
```

Included structured outputs:

- `InvoiceOutput`
- `SalesCallReport`
- `VideoAdReport`
- `RedactedText`
- `Classification`

## Runtimes

Global default:

```python
from aibackends import configure

configure(runtime="llamacpp", model="gemma4-e2b")
```

Per-task override:

```python
from aibackends.tasks import extract_invoice

result = extract_invoice("invoice.pdf", runtime="anthropic", model="claude-sonnet-4-5")
```

Supported runtimes:

- `llamacpp`
- `transformers`
- `ollama`
- `lmstudio`
- `anthropic`
- `together`
- `groq`

For local `transformers` models, prompt rendering is configurable:

```python
from aibackends import configure

configure(
    runtime="transformers",
    model="google/gemma-3-270m-it",
    prompt_format="auto",  # auto | chat_template | text
)
```

`prompt_format="auto"` uses an inline or file-based override first, then the tokenizer's own chat template, and finally plain text as a fallback. You can force a custom template with `chat_template="..."` or `chat_template_path="template.jinja"`.

## Workflows

```python
from pathlib import Path

from aibackends import configure
from aibackends.workflows import SalesCallAnalyser

configure(runtime="llamacpp", model="gemma4-e2b")

results = SalesCallAnalyser().run_batch(
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

## Examples

See `examples/README.md` for setup and usage.

Runnable core examples with bundled sample data:

- `basic_task.py`
- `basic_task_transformers.py`
- `summarize_text.py`
- `classify_text.py`
- `redact_text.py`
- `extract_custom_schema.py`
- `sales_call_report.py`
- `video_ad_report.py`
- `batch_processing.py`
- `custom_pipeline.py`

Framework integration examples:

- `langgraph_agent.py`
- `pydantic_ai_agent.py`
- `openai_agents_sdk.py`
- `crewai_agent.py`
- `agno_agent.py`
- `llamaindex_agent.py`

## CLI

```bash
aibackends task extract-invoice --input invoice.pdf
aibackends task redact-pii --input transcript.txt --backend gliner --labels email,phone_number,user_name
aibackends task classify --input doc.txt --labels invoice,contract,receipt
aibackends pull gemma4-e2b --runtime llamacpp
aibackends check transformers
```

## Repo layout

```text
src/aibackends/
  core/
  schemas/
  steps/
  tasks/
  workflows/
  cli.py
examples/
docs/
tests/
```

## Development

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest tests
python3 -m mypy src tests
ruff check .
```

See `CONTRIBUTING.md` for task, workflow, and runtime contribution guidelines.
# Usage

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

## Configure A Runtime

Use `configure()` to set global LLM/embedding defaults:

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

Override the runtime for one call:

```python
from aibackends.models import CLAUDE_SONNET_4_5
from aibackends.runtimes import ANTHROPIC
from aibackends.tasks import extract_invoice

result = extract_invoice("invoice.pdf", runtime=ANTHROPIC, model=CLAUDE_SONNET_4_5)
```

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
or `backend="openai-privacy"`.

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

## Agent Frameworks

AIBackends tasks and workflows are plain Python objects. Configure them once,
then wrap `task.run(...)` or `workflow.run(...)` with the adapter required by
your agent framework. This keeps pipelines independent of LangGraph,
pydantic-ai, OpenAI Agents SDK, CrewAI, Agno, LlamaIndex, or any future framework
you adopt.

- LangGraph: wrap `task.run` or `workflow.run` with `@tool`
- pydantic-ai: pass wrapper functions in `tools=[...]`
- OpenAI Agents SDK: wrap with `@function_tool`
- CrewAI: wrap with `@tool("Name")`
- Agno: wrap with `@tool`
- LlamaIndex: use `FunctionTool.from_defaults(fn=...)`

See `examples/` for runnable framework examples.
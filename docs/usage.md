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

configure(runtime="llamacpp", model="gemma4-e2b")
```

Override the runtime for one call:

```python
from aibackends.tasks import extract_invoice

result = extract_invoice("invoice.pdf", runtime="anthropic", model="claude-sonnet-4-5")
```

You can also load YAML:

```python
from aibackends import load_config

load_config("aibackends.yml")
```

For local `transformers` models, prompt rendering is configurable:

```python
configure(
    runtime="transformers",
    model="google/gemma-3-270m-it",
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

Tasks are also available through the task registry as `BaseTask` objects:

```python
from aibackends.tasks import create_task

task = create_task(
    "summarize",
    runtime="llamacpp",
    model="gemma4-e2b",
)
summary = task.run("notes.txt")
```

## Use Workflows

```python
from pathlib import Path

from aibackends.workflows import create_workflow

workflow = create_workflow(
    "sales-call",
    runtime="llamacpp",
    model="gemma4-e2b",
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

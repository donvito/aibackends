# Concepts

AIBackends uses a few precise terms so new capabilities can be added without
turning the project into a pile of special cases.

The main design goal is framework independence: tasks and workflows are plain
Python objects with stable interfaces. Agent integrations should wrap those
objects, not own the business logic. That makes it easier to switch frameworks or
use multiple frameworks in the same product.

## Task

A task is the user-facing unit of work: one call in, one typed result out. Task
implementations use the `BaseTask` interface with a `run(...)` method, and the
public function wrappers keep the simple direct-call API.

Examples:

- `extract_invoice(path)` returns `InvoiceOutput`
- `redact_pii(text, backend="gliner")` returns `RedactedText`
- `classify(text, labels=[...])` returns `Classification`
- `summarize(text)` returns `str`

Tasks live in `src/aibackends/tasks` and can be registered with `TASK_SPEC`.

## Runtime

A runtime is a general LLM or embedding executor. It implements the common
runtime contract:

```python
complete(messages, schema=...)
embed(text)
```

Use the word runtime for providers that can power normal LLM calls across many
tasks. Examples:

- `transformers`
- `llamacpp`
- `ollama`
- `lmstudio`
- `anthropic`
- `together`
- `groq`

Runtime modules live in `src/aibackends/core/runtimes` and export
`RUNTIME_SPEC`. The supported Python-facing runtime refs are exported from
`aibackends.runtimes`.

## Backend

A backend is a swappable implementation for one capability. It may use a model,
but it does not implement the general `complete()` / `embed()` runtime contract.

PII detection is the current example:

- `gliner` uses the `nvidia/gliner-pii` model and returns detected PII spans.
- `openai-privacy` uses the `openai/privacy-filter` model through a token
classification pipeline.

These are model-backed PII backends, not runtimes. They solve a specific
capability and return domain objects such as `PIIEntity`.

Current PII backend files live under `src/aibackends/backends/pii`.

## Model

A model is the artifact or profile used by a runtime or backend.

Examples:

- `google/gemma-4-E2B-it`
- `unsloth/gemma-4-E2B-it-GGUF`
- `nvidia/gliner-pii`
- `openai/privacy-filter`

Transformer model profiles live under `src/aibackends/models`. They can provide
aliases, Hugging Face model ids, chat templates, and generation defaults. User
configuration always takes precedence over profile defaults. The supported
Python-facing model refs are exported from `aibackends.models`.

## Workflow

A workflow is an optional orchestration layer built from reusable steps. It is
for multi-stage work such as ingesting a file, processing it, running an LLM
analysis step, validating the result, and running batches.

Workflow classes live in `src/aibackends/workflows` and can be registered with
`WORKFLOW_SPEC`. Use `create_workflow(SalesCallAnalyser, ...)` or instantiate the
workflow class directly to build a configured workflow instance. For dynamic
lookup by name, use `get_workflow(name)`.

Workflows should stay independent of agent frameworks. If an agent needs a
workflow, wrap `workflow.run(...)` as that framework's tool function instead of
rewriting the workflow inside the agent framework.
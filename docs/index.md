# AIBackends

AIBackends is a Python library of ready-made AI tasks and workflows that are not
tied to any one agent framework. Build a task or pipeline once, then plug it into
LangGraph, pydantic-ai, OpenAI Agents SDK, CrewAI, Agno, LlamaIndex, or your own
application code.

Extract invoices, redact PII, classify documents, summarize text, analyse sales
calls, and analyse video ads with typed results.

```python
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import ExtractInvoiceTask, create_task

task = create_task(ExtractInvoiceTask, runtime=LLAMACPP, model=GEMMA4_E2B)

result = task.run("invoice.pdf")
print(result.total)
```

## What To Read

- [Concepts](concepts.md): the framework vocabulary, especially runtime vs backend vs model.
- [Architecture](architecture.md): the package map, request flow, and extension seams.
- [Usage](usage.md): install, configure, call tasks, and use workflows.
- [CLI](cli.md): run tasks, pull models, and check runtimes from the terminal.
- [Extending](extending.md): add a runtime, model profile, backend, task, or workflow by adding focused files.
- [API Reference](api-reference/index.md): the public API groups.


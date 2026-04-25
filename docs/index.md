# AIBackends

AIBackends is a Python library of ready-made AI tasks and workflows that are not
tied to any one agent framework. Build a task or pipeline once, then plug it into
LangGraph, pydantic-ai, OpenAI Agents SDK, CrewAI, Agno, LlamaIndex, or your own
application code.

Extract invoices, redact PII, classify documents, summarize text, analyse sales
calls, and analyse video ads with typed results.

```python
from aibackends.tasks import create_task

task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")

result = task.run("invoice.pdf")
print(result.total)
```

## What To Read

- [Concepts](concepts.md): the framework vocabulary, especially runtime vs backend vs model.
- [Usage](usage.md): install, configure, call tasks, and use workflows.
- [CLI](cli.md): run tasks, pull models, and check runtimes from the terminal.
- [Extending](extending.md): add a runtime, model profile, backend, task, or workflow by adding focused files.
- [API Reference](api-reference/index.md): the public API groups.


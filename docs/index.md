# AIBackends

Run AI tasks and workflows locally.

AIBackends uses first-class `llamacpp` and `transformers` runtimes to build
extraction, classification, embeddings, redaction, and analysis pipelines in
plain Python.

```python
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import ExtractInvoiceTask, create_task

task = create_task(ExtractInvoiceTask, runtime=LLAMACPP, model=GEMMA4_E2B)

result = task.run("invoice.pdf")
print(result.total)
```

## What To Read

- [Concepts](concepts.md): the runtime, backend, model, and workflow vocabulary.
- [Architecture](architecture.md): the package map, request flow, and extension seams.
- [Usage](usage.md): install, configure local runtimes, call tasks, and use workflows.
- [CLI](cli.md): run tasks, pull models, and check runtimes from the terminal.
- [Extending](extending.md): add a runtime, model profile, backend, task, or workflow by adding focused files.
- [API Reference](api-reference/index.md): the public API groups.


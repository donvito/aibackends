# AIBackends

AIBackends is a Python library of ready-made AI tasks that plug into any agent
framework as tools. Extract invoices, redact PII, classify documents, summarize
text, analyse sales calls, and analyse video ads with one function call and a
typed result.

```python
from aibackends import configure
from aibackends.tasks import extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")

result = extract_invoice("invoice.pdf")
print(result.total)
```

## What To Read

- [Concepts](concepts.md): the framework vocabulary, especially runtime vs backend vs model.
- [Usage](usage.md): install, configure, call tasks, use workflows, and run the CLI.
- [Extending](extending.md): add a runtime, model profile, backend, task, or workflow by adding focused files.
- [API Reference](api-reference/index.md): the public API groups.


# Quickstart

Install the package:

```bash
pip install aibackends
pip install aibackends[llamacpp-metal]
pip install aibackends[pdf]
```

Configure a default runtime once:

```python
from aibackends import configure

configure(runtime="llamacpp", model="gemma4-e2b")
```

Call a task directly:

```python
from aibackends.tasks import extract_invoice, summarize

invoice = extract_invoice("invoice.pdf")
summary = summarize("meeting-notes.txt")
```

Override the runtime for one call:

```python
invoice = extract_invoice("invoice.pdf", runtime="anthropic", model="claude-sonnet-4-5")
```

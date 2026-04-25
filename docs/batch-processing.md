# Batch Processing

All workflows support concurrent batch execution.

```python
from pathlib import Path

from aibackends.workflows import InvoiceProcessor

result = InvoiceProcessor().run_batch(
    inputs=Path("./invoices").glob("*.pdf"),
    max_concurrency=4,
    on_error="collect",
)

print(result.results)
print(result.errors)
```

`on_error` supports:

- `"raise"`
- `"skip"`
- `"collect"`

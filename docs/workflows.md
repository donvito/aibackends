# Workflows

Workflows are optional orchestration layers built on reusable steps.

Included workflows:

- `InvoiceProcessor`
- `SalesCallAnalyser`
- `VideoAdIntelligence`
- `PIIRedactor`

Each workflow supports:

- `run(...)`
- `run_async(...)`
- `run_batch(...)`
- `run_batch_async(...)`

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

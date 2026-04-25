# Tasks

AIBackends ships with direct-call task functions under `aibackends.tasks`.

## Document tasks

- `extract_invoice(path)` -> `InvoiceOutput`
- `redact_pii(text, backend="gliner", labels=[...])` -> `RedactedText`
- `classify(text, labels=...)` -> `Classification`
- `extract(text, schema=...)` -> your Pydantic model
- `summarize(text)` -> `str`
- `embed(text)` -> `list[float]`

`redact_pii` uses the selected PII backend such as `gliner` or `openai-privacy` and does not read the global `configure()` runtime/model settings. When using `gliner`, you can provide custom entity labels with `labels=[...]`.

## Audio tasks

- `analyse_sales_call(path)` -> `SalesCallReport`

## Video tasks

- `analyse_video_ad(path)` -> `VideoAdReport`

Every task also exposes an async variant with the `_async` suffix.

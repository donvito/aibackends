# CLI Guide

AIBackends ships a small `aibackends` command-line tool for running individual
tasks, downloading local models, and sanity-checking runtimes.

The script is registered in `pyproject.toml`:

```toml
[project.scripts]
aibackends = "aibackends.cli:app"
```

The implementation lives in `src/aibackends/cli.py` and is built with
[Typer](https://typer.tiangolo.com/).

## Quick Start

```bash
aibackends --help
aibackends task --help
aibackends pull --help
aibackends check --help
```

If you have not installed the package globally you can run it via your
environment manager:

```bash
uv run aibackends task --help
python -m aibackends --help   # if your environment exposes it as a module
```

## Commands

### `aibackends task` — run a single task

General shape:

```bash
aibackends task <name> --input <path-or-text> \
  [--runtime <name>] \
  [--model <name>] \
  [--labels a,b,c] \
  [--backend <name>] \
  [--schema dotted.path.SchemaModel]
```

`--input` accepts either a file path or raw text. The CLI uses the same
`load_text_input(...)` helper as the Python API, so:

The CLI keeps runtime and model selection string-based (`--runtime`, `--model`)
and resolves those names at the boundary before calling the stricter Python API.

- A real file path (e.g. `./notes.txt`, `./invoice.pdf`) is read from disk.
- Anything else is treated as raw text.

Supported task names match the registry (run `aibackends task --help` or look
under `src/aibackends/tasks/`):

- `summarize`
- `extract`
- `classify`
- `embed`
- `redact-pii`
- `extract-invoice`
- `analyse-sales-call`
- `analyse-video-ad`

Per-task examples:

```bash
# summarize raw text
aibackends task summarize \
  --input "Meeting notes: Alice owns pricing, Bob owns analytics..." \
  --runtime llamacpp --model gemma4-e2b

# summarize a file
aibackends task summarize \
  --input examples/data/meeting_notes.txt \
  --runtime llamacpp --model gemma4-e2b

# classify with required labels
aibackends task classify \
  --input "I love this product" \
  --labels positive,negative,neutral \
  --runtime llamacpp --model gemma4-e2b

# extract-invoice (structured output)
aibackends task extract-invoice \
  --input examples/data/invoice.pdf \
  --runtime llamacpp --model gemma4-e2b

# analyse-sales-call (structured output)
aibackends task analyse-sales-call \
  --input examples/data/sales_call.txt \
  --runtime llamacpp --model gemma4-e2b

# redact-pii (no LLM; uses a PII backend)
aibackends task redact-pii \
  --input "Call me at 555-1234, john@example.com" \
  --backend gliner

# redact-pii with the local privacy-filter backend
aibackends task redact-pii \
  --input "Call me at 555-1234, john@example.com" \
  --backend openai-privacy

# extract with a custom Pydantic schema
aibackends task extract \
  --input "John Doe, 35, NYC" \
  --schema myproject.schemas.PersonSchema \
  --runtime llamacpp --model gemma4-e2b
```

Notes:

- `redact-pii` does not use the `--runtime` / `--model` flags. It dispatches to
  a PII backend such as `gliner` or `openai-privacy` (`privacy-filter`).
- `classify` requires `--labels`. `redact-pii` accepts `--labels` only when used
with the `gliner` backend (custom entity types).
- `extract` requires `--schema` pointing to a Pydantic model class via dotted
path (`package.module.SchemaName`).

### `aibackends pull` — pre-download a local model

```bash
aibackends pull gemma4-e2b --runtime llamacpp

# pin the cache directory
aibackends pull gemma4-e2b --runtime llamacpp --cache-dir ./.hf-cache
```

This invokes `ModelManager.pull_model(...)` and prints the resolved local
location. By default models go into the standard Hugging Face cache, usually
`~/.cache/huggingface/hub`.

### `aibackends check` — sanity-check a runtime

```bash
aibackends check llamacpp --model gemma4-e2b
aibackends check transformers
```

It instantiates the runtime client and prints the resolved class plus model
name. No tokens are spent and no LLM call is made.

## Output Format

`_serialize(...)` in `src/aibackends/cli.py` decides how each result is
printed:

- `pydantic.BaseModel` → pretty `model_dump_json(indent=2)`
- `dataclasses.dataclass` → `json.dumps(asdict(...), indent=2)`
- `pathlib.Path` → string path
- `str` → printed as-is (e.g. `summarize` returns plain text)
- anything else → `json.dumps(..., indent=2, default=str)`

So:

- `summarize` writes the summary text directly to stdout, ready to pipe into
another tool.
- Structured tasks (`extract-invoice`, `analyse-sales-call`,
`analyse-video-ad`, `classify`, `extract`, `redact-pii`) emit indented JSON
you can pipe into `jq`:
  ```bash
  aibackends task extract-invoice --input invoice.pdf | jq '.total'
  ```

The JSON shape comes from each task's Pydantic output model. Useful field
names you'll likely pipe with `jq`:

- `redact-pii` → `RedactedText`: `original_text`, `redacted_text`,
`entities_found`, `redaction_map`, `backend_used`
- `extract-invoice` → `InvoiceOutput`: `vendor`, `invoice_number`, `total`,
`line_items`, ...
- `classify` → `Classification`: `label`, `confidence`
- `analyse-sales-call` → `SalesCallReport`
- `analyse-video-ad` → `VideoAdReport`

## Recipes

### Summarize a resume (PDF)

PDFs are read automatically when the `pdf` extra is installed:

```bash
pip install 'aibackends[pdf]'

aibackends task summarize \
  --input ./resume.pdf \
  --runtime llamacpp --model gemma4-e2b
```

`summarize` writes plain text to stdout, so saving the summary is a redirect:

```bash
aibackends task summarize \
  --input ./resume.pdf \
  --runtime llamacpp --model gemma4-e2b \
  > resume_summary.txt
```

### Redact PII from a resume, then summarize

Two-step pipeline — `redact-pii` extracts the PDF text, scrubs PII, and emits
JSON; `jq` pulls out the redacted text; `summarize` summarizes it.

```bash
aibackends task redact-pii \
  --input ./resume.pdf \
  --backend gliner \
  | jq -r '.redacted_text' \
  > resume_redacted.txt

aibackends task summarize \
  --input "$(cat resume_redacted.txt)" \
  --runtime llamacpp --model gemma4-e2b
```

Notes:

- The field is `.redacted_text` (matches the `RedactedText` schema), not
`.redacted`.
- The first run with `--backend gliner` downloads the GLiNER model from
Hugging Face. Behind a corporate proxy, set `HTTPS_PROXY` /
`HF_ENDPOINT` accordingly.
- Image-only / scanned PDFs have no text layer. PyMuPDF returns empty text
and there is nothing to redact or summarize. AIBackends does not OCR;
preprocess with a tool like `ocrmypdf` first.

For a single-process Python equivalent of the same recipe (one pipeline
run, no shell pipe and no intermediate file), see
`[examples/workflows/resume_redact_summarize.py](https://github.com/donvito/aibackends/blob/main/examples/workflows/resume_redact_summarize.py)`.

### Extract structured fields from a resume

`extract` accepts a custom Pydantic schema by dotted import path:

```bash
aibackends task extract \
  --input ./resume.pdf \
  --schema myproject.schemas.ResumeProfile \
  --runtime llamacpp --model gemma4-e2b
```

The schema module must be importable from the current Python environment.

## What The CLI Does Not Cover

The CLI is intentionally focused on single-task execution. The following are
Python-API-only today:

- Workflows (`invoice`, `pii-redactor`, `sales-call`, `video-ad`) — use
`create_workflow(WorkflowClass, ...)` from `aibackends.workflows`.
- Batch processing (`workflow.run_batch(...)`).
- Custom pipelines built on `Pipeline` / `BaseStep`.

See [Usage](usage.md) and the runnable examples under `examples/workflows/`
for these patterns.
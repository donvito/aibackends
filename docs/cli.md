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
  a PII backend such as `gliner` or `openai-privacy`.
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
aibackends check ollama --model llama3
aibackends check anthropic --model claude-3-5-sonnet-latest
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

## What The CLI Does Not Cover

The CLI is intentionally focused on single-task execution. The following are
Python-API-only today:

- Workflows (`invoice`, `pii-redactor`, `sales-call`, `video-ad`) — use
  `create_workflow(...)` from `aibackends.workflows`.
- Batch processing (`workflow.run_batch(...)`).
- Custom pipelines built on `Pipeline` / `BaseStep`.

See [Usage](usage.md) and the runnable examples under `examples/workflows/`
for these patterns.

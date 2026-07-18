# Benchmarks

Scripts that measure the effect of model caching (runtime reuse, PII backend
caches) and task latency per model, writing markdown reports to
`benchmarks/reports/`. Reports are dated and intended to be committed so
results can be referenced on GitHub.

LLM benchmarks only accept models from the recommended `aibackends.models`
catalog (`gemma4-e2b`, `gemma3-270m-it`, `minilm-l6`, ...), so committed
numbers stay comparable across runs and machines.

## Running One At A Time

Benchmarks share one GPU, so they must not run concurrently. Two safeguards
enforce this:

- A lock file (`benchmarks/.benchmark.lock`) makes any benchmark exit
  immediately if another one is already running. Delete the file if a
  crashed run leaves it behind.
- `run_all.py` runs every benchmark sequentially, each in its own process,
  so one model's RAM/VRAM is fully released before the next starts:

```bash
python benchmarks/run_all.py --runtime transformers --warm-calls 10
python benchmarks/run_all.py --skip pii --warm-calls 100
```

## Scripts

### `benchmark_runtime_reuse.py`

Compares a fresh runtime per call (`reuse_runtime=False`) against the
process-wide runtime cache and `preload()`.

```bash
# Smallest catalog chat model, quick run (transformers)
python benchmarks/benchmark_runtime_reuse.py \
    --runtime transformers --model gemma3-270m-it --warm-calls 5

# llama.cpp with a catalog GGUF model
python benchmarks/benchmark_runtime_reuse.py \
    --runtime llamacpp --model gemma4-e2b
```

Requires `aibackends[transformers]` or `aibackends[llamacpp]`.

### `benchmark_tasks.py`

Per-model task latency: loads each model once via `preload()`, then times
warm `summarize` / `classify` / `extract` calls per model, `embed` against
embedding models, and a VL describe-image call against VL models (llama.cpp
only, Gemma / Qwen VL). Pass `all` to any model list to benchmark every
recommended catalog model in that category.

```bash
python benchmarks/benchmark_tasks.py --runtime transformers \
    --models gemma3-270m-it --embed-models minilm-l6 --warm-calls 10

# Every recommended chat, embedding, and VL model on llama.cpp
python benchmarks/benchmark_tasks.py --runtime llamacpp \
    --models all --embed-models all --vl-models all
```

Requires `aibackends[transformers]` or `aibackends[llamacpp]`.

### `benchmark_pii_backends.py`

Measures the one-time PII model load against warm `redact()` calls.

```bash
python benchmarks/benchmark_pii_backends.py --backend gliner
```

Requires `aibackends[pii]`.

## Consistency And Degradation

Sample counts are configurable with `--warm-calls` (default 10). With 10 or
more samples the reports include a consistency section: standard deviation,
coefficient of variation, p50/p95, and a drift figure comparing the mean of
the last quarter of calls against the first quarter (positive drift means
calls slowed down over the run). A per-segment mean table shows the trend
across the run so inference degradation is visible at a glance. Raise
`--warm-calls` (e.g. to 100) for a longer soak when checking degradation.

## Reports

Reports land in `benchmarks/reports/` as `YYYY-MM-DD_<benchmark>.md` and
include the environment (platform, Python, package versions) alongside the
timing table, so numbers stay comparable across machines and versions.
Re-running a benchmark on the same day overwrites that day's report.

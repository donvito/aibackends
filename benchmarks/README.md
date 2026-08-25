# Benchmarks

Scripts that measure the effect of model caching (runtime reuse, PII and
moderation backend caches) and task latency per model, writing markdown reports to
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
python benchmarks/run_all.py --skip pii guardrails --warm-calls 100
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

# LiquidAI LFM2.5-VL-3B image latency on CPU (Q4_K_M profile default)
python benchmarks/benchmark_tasks.py --runtime llamacpp --device cpu \
    --tasks vl --vl-models lfm2.5-vl-3b
```

`--device cpu` (or `gpu`) forces the device instead of auto-detecting the
hardware. Forced-device runs get their own report file
(`tasks-<runtime>-<device>`), so CPU and GPU numbers can be committed side
by side. GGUF quantization follows the model profile (for example both LFM2.5
profiles default to `Q4_K_M`) unless overridden in the config.

Requires `aibackends[transformers]` or `aibackends[llamacpp]`.

### `benchmark_pii_backends.py`

Measures the one-time PII model load against warm `redact()` calls.

```bash
python benchmarks/benchmark_pii_backends.py --backend gliner
```

Requires `aibackends[pii]`.

### `benchmark_gliguard_cpu.py`

Forces GliGuard onto CPU and measures the first moderation call including
model loading, explicit `backend.load()`, warm prompt/response moderation, and
native prompt/response batch throughput.

```bash
python benchmarks/benchmark_gliguard_cpu.py \
    --warm-calls 10 --batch-size 8
```

The report includes total batch latency, derived per-item latency, and
items/second. The model must already be downloaded if you want the first-call
number to exclude network transfer.

Requires `aibackends[guardrails]`.

### `benchmark_gliner25_cpu.py`

Compares the three GLiNER2.5 boundary checkpoints on CPU: local model
construction, warm entity extraction, long-document extraction, constrained
classification, Joint IE, combined-schema extraction, and native entity batch
throughput. Every scenario runs through the aibackends `gliner25` backend.

```bash
python3 benchmarks/benchmark_gliner25_cpu.py \
    --models small base multi --warm-calls 10 --batch-size 8
```

Models run sequentially so only one checkpoint is resident at a time. Run the
applied accuracy eval separately before using latency alone to choose a model:

```bash
python3 evals/eval_gliner25.py --models small base multi --device cpu
```

Requires `aibackends[gliner2]`. Pre-download the checkpoints if model load
timings should exclude network transfer.
The latest committed comparison is
[`2026-08-25_gliner25-cpu.md`](reports/2026-08-25_gliner25-cpu.md).

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

# Task Benchmark

Runtime `llamacpp` (device: cpu), 10 warm calls per task, max_tokens 128. Model load is paid once per model via `preload()`; task calls below run against the warm runtime.

## Environment

- Date: 2026-08-15 09:41:28 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64
- GPU: none detected
- Python: 3.12.3
- aibackends: 0.3.0
- transformers: not installed
- torch: not installed
- llama-cpp-python: 0.3.34

## Results

### `lfm2.5-vl-3b` (VL)

Prompt: 'Describe this receipt in one sentence.' with `receipt1.jpeg`.

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First VL call (includes model load) | 1 | 23,383.7 | 23,383.7 | 23,383.7 |
| VL describe-image (warm) | 10 | 11,095.1 | 10,602.4 | 11,630.0 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| VL describe-image (warm) | 348.5 | 3.1% | 10,965.3 | 11,630.0 | -0.1% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, VL describe-image (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 11,378 | 11,259 | 10,602 | 10,765 | 10,845 | 11,579 | 10,952 | 10,965 | 11,630 | 10,975 |

## Notes

- Structured tasks (`classify`, `extract`) include JSON parsing and
  may retry on validation failures, so their timings can exceed a
  single `complete()` call.
- VL (image) inputs are supported by the llama.cpp runtime for
  Gemma, Qwen VL, and LiquidAI LFM VL GGUF models only.
- Only models from the `aibackends.models` catalog are benchmarked.

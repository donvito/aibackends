# Task Benchmark

Runtime `transformers`, 10 warm calls per task, max_tokens 128. Model load is paid once per model via `preload()`; task calls below run against the warm runtime.

## Environment

- Date: 2026-07-18 10:34:56 UTC
- Platform: Windows-11-10.0.26200-SP0
- Processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- Python: 3.13.12
- aibackends: 0.2.1
- transformers: 5.1.0
- torch: 2.11.0
- llama-cpp-python: not installed

## Results

### `gemma3-270m-it`

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Model load (`preload()`) | 1 | 11,079.6 | 11,079.6 | 11,079.6 |
| `summarize` (warm) | 10 | 7,713.9 | 7,175.7 | 9,019.5 |
| `classify` (warm) | 10 | 9,203.9 | 8,614.1 | 9,957.1 |
| `extract` (warm) | 10 | 8,373.1 | 7,657.3 | 9,718.9 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| `summarize` (warm) | 541.3 | 7.0% | 7,427.0 | 9,019.5 | +15.3% |
| `classify` (warm) | 468.8 | 5.1% | 8,973.6 | 9,957.1 | +5.1% |
| `extract` (warm) | 694.5 | 8.3% | 7,995.8 | 9,718.9 | -14.1% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, `summarize` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 7,395 | 7,410 | 7,427 | 7,969 | 7,524 | 7,176 | 7,338 | 7,835 | 9,019 | 8,046 |

Per-segment mean (ms) across the run, `classify` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 8,869 | 8,905 | 9,175 | 8,974 | 9,957 | 9,808 | 9,059 | 8,614 | 8,900 | 9,778 |

Per-segment mean (ms) across the run, `extract` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 8,749 | 9,719 | 9,352 | 8,313 | 8,339 | 7,930 | 7,657 | 7,811 | 7,996 | 7,865 |

### `minilm-l6` (embeddings)

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First `embed` (includes model load) | 1 | 1,734.1 | 1,734.1 | 1,734.1 |
| `embed` (warm) | 10 | 7.1 | 6.5 | 7.6 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| `embed` (warm) | 0.3 | 4.9% | 7.2 | 7.6 | +4.4% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, `embed` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 8 | 7 |

### VL

Skipped: image inputs require the `llamacpp` runtime.

## Notes

- Structured tasks (`classify`, `extract`) include JSON parsing and
  may retry on validation failures, so their timings can exceed a
  single `complete()` call.
- VL (image) inputs are supported by the llama.cpp runtime for
  Gemma and Qwen VL GGUF models only.
- Only models from the `aibackends.models` catalog are benchmarked.

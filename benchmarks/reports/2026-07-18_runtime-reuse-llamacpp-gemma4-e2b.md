# Runtime Reuse Benchmark

Runtime `llamacpp`, model `gemma4-e2b`, max_tokens 16, prompt: 'Reply with a short greeting.'

## Environment

- Date: 2026-07-18 10:50:57 UTC
- Platform: Windows-11-10.0.26200-SP0
- Processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- GPU: NVIDIA GeForce RTX 3090
- Python: 3.13.12
- aibackends: 0.2.1
- transformers: 5.1.0
- torch: 2.11.0+cu128
- llama-cpp-python: 0.3.34

## Results

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Fresh runtime per call (`reuse_runtime=False`) | 3 | 2,758.2 | 2,453.2 | 3,349.4 |
| First call with reuse (load + inference) | 1 | 2,438.9 | 2,438.9 | 2,438.9 |
| Warm calls with reuse | 10 | 73.2 | 66.7 | 105.1 |
| `preload()` (load only) | 1 | 2,239.0 | 2,239.0 | 2,239.0 |
| First call after `preload()` | 1 | 140.8 | 140.8 | 140.8 |

Warm calls are **37.7x** faster than fresh-runtime calls (mean over mean).

## Consistency

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| Warm calls with reuse | 12.4 | 17.0% | 67.6 | 105.1 | -28.1% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, Warm calls with reuse:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 105 | 84 | 69 | 67 | 67 | 67 | 68 | 69 | 69 | 68 |

## Notes

- Fresh-runtime calls still benefit from the OS file cache after the
  first load, so a true cold process start is slower than shown.
- Warm calls measure inference only; the model load cost is paid once
  per process (or explicitly via `preload()`).

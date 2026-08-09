# Runtime Reuse Benchmark

Runtime `llamacpp`, model `lfm2.5-2.6b`, max_tokens 16, prompt: 'Reply with a short greeting.'

## Environment

- Date: 2026-08-09 06:43:35 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64
- GPU: none detected
- Python: 3.12.3
- aibackends: 0.3.0
- transformers: 5.14.1
- torch: 2.13.0+cpu
- llama-cpp-python: 0.3.34

## Results

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Fresh runtime per call (`reuse_runtime=False`) | 3 | 4,484.7 | 3,664.9 | 5,871.3 |
| First call with reuse (load + inference) | 1 | 3,922.3 | 3,922.3 | 3,922.3 |
| Warm calls with reuse | 10 | 2,479.5 | 2,436.3 | 2,586.9 |
| `preload()` (load only) | 1 | 1,912.9 | 1,912.9 | 1,912.9 |
| First call after `preload()` | 1 | 2,201.0 | 2,201.0 | 2,201.0 |

Warm calls are **1.8x** faster than fresh-runtime calls (mean over mean).

## Consistency

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| Warm calls with reuse | 45.9 | 1.8% | 2,454.3 | 2,586.9 | +0.4% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, Warm calls with reuse:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 2,454 | 2,447 | 2,447 | 2,436 | 2,501 | 2,505 | 2,587 | 2,500 | 2,476 | 2,443 |

## Notes

- Fresh-runtime calls still benefit from the OS file cache after the
  first load, so a true cold process start is slower than shown.
- Warm calls measure inference only; the model load cost is paid once
  per process (or explicitly via `preload()`).

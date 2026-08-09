# Runtime Reuse Benchmark

Runtime `transformers`, model `lfm2.5-2.6b`, max_tokens 16, prompt: 'Reply with a short greeting.'

## Environment

- Date: 2026-08-09 06:44:35 UTC
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
| Fresh runtime per call (`reuse_runtime=False`) | 3 | 5,721.9 | 4,647.2 | 7,787.1 |
| First call with reuse (load + inference) | 1 | 4,606.9 | 4,606.9 | 4,606.9 |
| Warm calls with reuse | 10 | 3,058.5 | 2,898.0 | 3,738.7 |
| `preload()` (load only) | 1 | 1,871.4 | 1,871.4 | 1,871.4 |
| First call after `preload()` | 1 | 3,173.6 | 3,173.6 | 3,173.6 |

Warm calls are **1.9x** faster than fresh-runtime calls (mean over mean).

## Consistency

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| Warm calls with reuse | 257.7 | 8.4% | 2,974.4 | 3,738.7 | +17.0% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, Warm calls with reuse:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 2,988 | 2,977 | 2,974 | 2,946 | 2,919 | 2,923 | 2,898 | 2,979 | 3,242 | 3,739 |

## Notes

- Fresh-runtime calls still benefit from the OS file cache after the
  first load, so a true cold process start is slower than shown.
- Warm calls measure inference only; the model load cost is paid once
  per process (or explicitly via `preload()`).

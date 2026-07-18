# Runtime Reuse Benchmark

Runtime `transformers`, model `gemma3-270m-it`, max_tokens 16, prompt: 'Reply with a short greeting.'

## Environment

- Date: 2026-07-18 10:34:55 UTC
- Platform: Windows-11-10.0.26200-SP0
- Processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- Python: 3.13.12
- aibackends: 0.2.1
- transformers: 5.1.0
- torch: 2.11.0
- llama-cpp-python: not installed

## Results

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Fresh runtime per call (`reuse_runtime=False`) | 3 | 10,078.6 | 6,320.3 | 16,158.3 |
| First call with reuse (load + inference) | 1 | 5,972.4 | 5,972.4 | 5,972.4 |
| Warm calls with reuse | 10 | 909.3 | 863.6 | 969.3 |
| `preload()` (load only) | 1 | 4,627.7 | 4,627.7 | 4,627.7 |
| First call after `preload()` | 1 | 1,032.8 | 1,032.8 | 1,032.8 |

Warm calls are **11.1x** faster than fresh-runtime calls (mean over mean).

## Consistency

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| Warm calls with reuse | 35.4 | 3.9% | 907.5 | 969.3 | +6.5% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, Warm calls with reuse:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 875 | 864 | 911 | 951 | 908 | 969 | 898 | 866 | 933 | 918 |

## Notes

- Fresh-runtime calls still benefit from the OS file cache after the
  first load, so a true cold process start is slower than shown.
- Warm calls measure inference only; the model load cost is paid once
  per process (or explicitly via `preload()`).

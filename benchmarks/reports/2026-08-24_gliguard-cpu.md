# GliGuard CPU Benchmark

Backend `gliguard` (model `fastino/gliguard-LLMGuardrails-300M`), device forced to `cpu`, 10 timed samples per warm scenario.

## Environment

- Date: 2026-08-24 17:19:25 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64
- GPU: none detected
- Python: 3.12.3
- aibackends: 0.4.0
- gliner2: 2.0.0
- transformers: 4.57.6
- torch: 2.13.0
- protobuf: 7.36.0
- Logical CPUs: 4
- Torch CPU threads: 4

## Results

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First prompt moderation (load + inference) | 1 | 6,053.9 | 6,053.9 | 6,053.9 |
| `backend.load(device="cpu")` | 1 | 3,073.8 | 3,073.8 | 3,073.8 |
| Warm prompt moderation | 10 | 85.1 | 83.4 | 87.4 |
| Warm response moderation | 10 | 72.2 | 71.0 | 75.0 |
| Prompt batch (size 8) | 10 | 517.3 | 506.6 | 532.8 |
| Response batch (size 8) | 10 | 333.0 | 327.0 | 339.4 |

Warm prompt moderation is **71.1x** faster than the first prompt call that includes model loading.

## Batch Throughput

| Scenario | Batch size | Mean batch (ms) | Mean/item (ms) | Items/s |
|---|---|---|---|---|
| Prompt moderation | 8 | 517.3 | 64.7 | 15.5 |
| Response moderation | 8 | 333.0 | 41.6 | 24.0 |

## Consistency

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| Warm prompt moderation | 1.1 | 1.3% | 85.0 | 87.4 | -0.7% |
| Warm response moderation | 1.1 | 1.6% | 71.9 | 75.0 | -1.1% |
| Prompt batch (size 8) | 9.6 | 1.9% | 514.4 | 532.8 | -1.8% |
| Response batch (size 8) | 5.2 | 1.6% | 333.4 | 339.4 | +1.0% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, Warm prompt moderation:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 85 | 85 | 86 | 85 | 84 | 87 | 86 | 84 | 85 | 83 |

Per-segment mean (ms) across the run, Warm response moderation:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 72 | 72 | 73 | 75 | 72 | 72 | 71 | 73 | 71 | 71 |

## Notes

- The benchmark forces `device="cpu"`; no GPU inference path is used.
- One untimed inference per schema is run before collecting warm samples.
- First-call timings include Python model construction from the local
  Hugging Face cache; a first-ever network download is not measured.
- Batch rows report total batch latency. The throughput table derives
  per-item latency and items/second from each mean batch latency.
- These timings measure performance, not moderation accuracy.

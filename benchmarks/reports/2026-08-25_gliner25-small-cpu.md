# GLiNER 2.5 CPU Benchmark

Backend `gliner25`, model `gliner25-small`, device forced to `cpu`, 10 timed samples per warm scenario.

## Environment

- Date: 2026-08-25 09:05:03 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64
- GPU: none detected
- Python: 3.12.3
- aibackends: 0.6.0
- gliner2: 2.0.0
- transformers: 4.57.6
- torch: 2.13.0
- protobuf: 7.36.0
- Logical CPUs: 4
- Torch CPU threads: 4

## Results

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First entity extraction (load + inference) | 1 | 5,335.4 | 5,335.4 | 5,335.4 |
| `backend.load(device="cpu")` | 1 | 2,095.0 | 2,095.0 | 2,095.0 |
| Warm entity extraction | 10 | 20.5 | 20.3 | 20.9 |
| Warm constrained classification | 10 | 20.2 | 19.9 | 20.6 |
| Warm joint IE | 10 | 64.9 | 64.7 | 65.2 |
| Warm long-document extraction | 10 | 208.5 | 203.1 | 213.6 |
| Native NER batch (size 8) | 10 | 47.6 | 46.5 | 51.3 |

Warm entity extraction is **260.4x** faster than the first call that includes model loading.

## Native batch throughput

| Scenario | Batch size | Mean batch (ms) | Mean/item (ms) | Items/s |
|---|---|---|---|---|
| Entity extraction | 8 | 47.6 | 6.0 | 168.0 |

## Consistency

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| Warm entity extraction | 0.2 | 1.0% | 20.4 | 20.9 | -2.4% |
| Warm constrained classification | 0.2 | 0.9% | 20.2 | 20.6 | -1.9% |
| Warm joint IE | 0.2 | 0.2% | 64.9 | 65.2 | -0.3% |
| Warm long-document extraction | 3.5 | 1.7% | 209.1 | 213.6 | -2.8% |
| Native NER batch (size 8) | 1.4 | 2.8% | 47.3 | 51.3 | -4.0% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, Warm entity extraction:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 21 | 21 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |

Per-segment mean (ms) across the run, Warm constrained classification:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 21 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 | 20 |

## Notes

- The benchmark forces `device="cpu"`; no GPU inference path is used.
- One untimed inference per scenario is run before collecting warm samples.
- First-call timings include Python model construction from the local
  Hugging Face cache; a first-ever network download is not measured.
- Constrained classification and joint IE reuse the loaded extractor.
- Batch rows report total batch latency. The throughput table derives
  per-item latency and items/second from each mean batch latency.
- These timings measure performance, not extraction accuracy.

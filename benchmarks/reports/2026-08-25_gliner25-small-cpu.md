# GLiNER 2.5 CPU Benchmark

Backend `gliner25`, model `gliner25-small`, device forced to `cpu`, 10 timed samples per warm scenario.

## Environment

- Date: 2026-08-25 08:54:44 UTC
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
| First entity extraction (load + inference) | 1 | 4,859.1 | 4,859.1 | 4,859.1 |
| `backend.load(device="cpu")` | 1 | 2,020.1 | 2,020.1 | 2,020.1 |
| Warm entity extraction | 10 | 21.0 | 20.7 | 22.2 |
| Warm constrained classification | 10 | 20.4 | 20.3 | 20.6 |
| Warm joint IE | 10 | 65.4 | 65.1 | 65.6 |
| Warm long-document extraction | 10 | 203.2 | 199.6 | 206.8 |

Warm entity extraction is **231.9x** faster than the first call that includes model loading.

## Consistency

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| Warm entity extraction | 0.5 | 2.2% | 20.8 | 22.2 | +2.7% |
| Warm constrained classification | 0.1 | 0.4% | 20.4 | 20.6 | -0.6% |
| Warm joint IE | 0.2 | 0.2% | 65.5 | 65.6 | +0.1% |
| Warm long-document extraction | 2.7 | 1.3% | 203.1 | 206.8 | -1.1% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, Warm entity extraction:

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 21 | 21 | 21 | 21 | 21 | 21 | 21 | 21 | 22 | 21 |

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
- These timings measure performance, not extraction accuracy.

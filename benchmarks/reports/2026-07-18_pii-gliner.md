# PII Backend Benchmark

Backend `gliner` (model `nvidia/gliner-pii`), sample text of 163 characters.

## Environment

- Date: 2026-07-18 10:39:47 UTC
- Platform: Windows-11-10.0.26200-SP0
- Processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- Python: 3.13.12
- aibackends: 0.2.1
- gliner: 0.2.26
- transformers: 5.1.0
- torch: 2.11.0

## Results

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First `redact()` including load | 1 | 13,387.8 | 13,387.8 | 13,387.8 |
| `backend.load()` (cold) | 1 | 5,269.6 | 5,269.6 | 5,269.6 |
| Warm `redact()` calls | 10 | 454.4 | 413.9 | 479.5 |

Warm redactions are **29.5x** faster than the first call that includes the model load.

## Notes

- The model is cached process-wide after the first call; use
  `backend.load()` at startup to move the load cost out of the first
  redaction.

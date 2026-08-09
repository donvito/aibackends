# Task Benchmark

Runtime `transformers`, 5 warm calls per task, max_tokens 512. Model load is paid once per model via `preload()`; task calls below run against the warm runtime.

## Environment

- Date: 2026-08-09 06:50:14 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64
- GPU: none detected
- Python: 3.12.3
- aibackends: 0.3.0
- transformers: 5.14.1
- torch: 2.13.0+cpu
- llama-cpp-python: 0.3.34

## Results

### `lfm2.5-2.6b`

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Model load (`preload()`) | 1 | 3,816.2 | 3,816.2 | 3,816.2 |
| `summarize` (warm) | 5 | 42,553.4 | 41,159.9 | 43,148.5 |
| `classify` (warm) | 5 | 140,859.4 | 91,295.2 | 260,710.7 |
| `extract` (warm) | 5 | 23,713.1 | 21,331.7 | 26,764.1 |

## Notes

- Structured tasks (`classify`, `extract`) include JSON parsing and
  may retry on validation failures, so their timings can exceed a
  single `complete()` call.
- VL (image) inputs are supported by the llama.cpp runtime for
  Gemma and Qwen VL GGUF models only.
- Only models from the `aibackends.models` catalog are benchmarked.

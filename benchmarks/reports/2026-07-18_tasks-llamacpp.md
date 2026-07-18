# Task Benchmark

Runtime `llamacpp`, 10 warm calls per task, max_tokens 128. Model load is paid once per model via `preload()`; task calls below run against the warm runtime.

## Environment

- Date: 2026-07-18 10:51:00 UTC
- Platform: Windows-11-10.0.26200-SP0
- Processor: AMD64 Family 25 Model 33 Stepping 2, AuthenticAMD
- GPU: NVIDIA GeForce RTX 3090
- Python: 3.13.12
- aibackends: 0.2.1
- transformers: 5.1.0
- torch: 2.11.0+cu128
- llama-cpp-python: 0.3.34

## Results

### `gemma3-12b`

Model load failed: 404 Client Error. (Request ID: Root=1-6a5b5a95-3746b4f35d1bfb2c1b3b663a;c6c865fa-308a-4138-b851-e3ad56573877)

Repository Not Found for url: https://huggingface.co/api/models/bartowski/gemma-3-12b-it-GGUF/tree/main?recursive=true&expand=false.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated and your token has the required permissions.
For more details, see https://huggingface.co/docs/huggingface_hub/authentication

### `gemma3-4b`

Model load failed: 404 Client Error. (Request ID: Root=1-6a5b5a95-1a861d7657ba673669d43d45;7ae2513d-c3b5-4083-9365-c6bc834c28d4)

Repository Not Found for url: https://huggingface.co/api/models/bartowski/gemma-3-4b-it-GGUF/tree/main?recursive=true&expand=false.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated and your token has the required permissions.
For more details, see https://huggingface.co/docs/huggingface_hub/authentication

### `gemma4-e2b`

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Model load (`preload()`) | 1 | 2,329.5 | 2,329.5 | 2,329.5 |
| `summarize` (warm) | 10 | 70.8 | 48.8 | 253.7 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| `summarize` (warm) | 64.3 | 90.8% | 49.6 | 253.7 | -67.6% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, `summarize` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 254 | 49 | 53 | 55 | 51 | 50 | 50 | 49 | 49 | 49 |

Failures:
- `classify` failed: classify could not produce valid structured output after 3 attempts.
- `extract` failed: extract could not produce valid structured output after 3 attempts.

### `gemma4-e4b`

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Model load (`preload()`) | 1 | 4,596.4 | 4,596.4 | 4,596.4 |
| `summarize` (warm) | 10 | 865.1 | 677.3 | 960.3 |
| `extract` (warm) | 10 | 8,750.0 | 8,165.3 | 9,234.8 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| `summarize` (warm) | 105.5 | 12.2% | 921.0 | 960.3 | -11.4% |
| `extract` (warm) | 348.2 | 4.0% | 8,758.9 | 9,234.8 | +1.3% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, `summarize` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 956 | 958 | 960 | 929 | 786 | 677 | 767 | 921 | 761 | 935 |

Per-segment mean (ms) across the run, `extract` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 9,235 | 8,578 | 8,759 | 8,165 | 8,402 | 8,414 | 8,983 | 8,923 | 8,918 | 9,123 |

Failures:
- `classify` failed: classify could not produce valid structured output after 3 attempts.

### `llama3-8b`

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Model load (`preload()`) | 1 | 71,750.5 | 71,750.5 | 71,750.5 |
| `summarize` (warm) | 10 | 863.3 | 760.3 | 986.8 |
| `classify` (warm) | 10 | 2,325.5 | 2,020.1 | 3,272.2 |
| `extract` (warm) | 10 | 1,243.3 | 1,034.1 | 1,554.1 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| `summarize` (warm) | 65.9 | 7.6% | 853.7 | 986.8 | -12.4% |
| `classify` (warm) | 359.1 | 15.4% | 2,165.1 | 3,272.2 | +38.0% |
| `extract` (warm) | 184.1 | 14.8% | 1,174.9 | 1,554.1 | -16.3% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, `summarize` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 987 | 925 | 841 | 804 | 760 | 854 | 892 | 897 | 868 | 806 |

Per-segment mean (ms) across the run, `classify` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 2,148 | 2,020 | 2,119 | 2,232 | 2,262 | 2,155 | 2,165 | 2,403 | 3,272 | 2,478 |

Per-segment mean (ms) across the run, `extract` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 1,218 | 1,446 | 1,554 | 1,367 | 1,385 | 1,156 | 1,034 | 1,044 | 1,054 | 1,175 |

### `mistral-7b`

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Model load (`preload()`) | 1 | 79,191.2 | 79,191.2 | 79,191.2 |
| `summarize` (warm) | 10 | 912.2 | 879.1 | 992.1 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| `summarize` (warm) | 35.2 | 3.9% | 898.5 | 992.1 | -8.0% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, `summarize` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 992 | 931 | 937 | 881 | 924 | 898 | 882 | 906 | 890 | 879 |

Failures:
- `classify` failed: classify could not produce valid structured output after 3 attempts.
- `extract` failed: extract could not produce valid structured output after 3 attempts.

### `phi4-mini`

Model load failed: 404 Client Error. (Request ID: Root=1-6a5b5bfa-1b15e8af22c59d335f850b69;07c4fd03-c044-4d03-a96b-2220cdaf1f88)

Repository Not Found for url: https://huggingface.co/api/models/bartowski/phi-4-mini-instruct-GGUF/tree/main?recursive=true&expand=false.
Please make sure you specified the correct `repo_id` and `repo_type`.
If you are trying to access a private or gated repo, make sure you are authenticated and your token has the required permissions.
For more details, see https://huggingface.co/docs/huggingface_hub/authentication

### `qwen3-vl-4b`

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Model load (`preload()`) | 1 | 2,828.1 | 2,828.1 | 2,828.1 |
| `summarize` (warm) | 10 | 373.0 | 341.6 | 477.0 |
| `classify` (warm) | 10 | 3,171.0 | 2,954.5 | 3,314.9 |
| `extract` (warm) | 10 | 1,316.7 | 1,148.0 | 1,641.1 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| `summarize` (warm) | 38.4 | 10.3% | 366.0 | 477.0 | -9.9% |
| `classify` (warm) | 106.8 | 3.4% | 3,179.9 | 3,314.9 | +2.5% |
| `extract` (warm) | 176.1 | 13.4% | 1,231.6 | 1,641.1 | -13.7% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, `summarize` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 477 | 352 | 342 | 366 | 367 | 352 | 376 | 352 | 376 | 371 |

Per-segment mean (ms) across the run, `classify` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 2,954 | 3,180 | 3,107 | 3,123 | 3,219 | 3,282 | 3,242 | 3,315 | 3,205 | 3,083 |

Per-segment mean (ms) across the run, `extract` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 1,229 | 1,641 | 1,620 | 1,349 | 1,289 | 1,232 | 1,183 | 1,148 | 1,181 | 1,296 |

### `qwen3-vl-8b`

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| Model load (`preload()`) | 1 | 90,924.6 | 90,924.6 | 90,924.6 |
| `summarize` (warm) | 10 | 535.6 | 512.0 | 619.8 |
| `classify` (warm) | 10 | 2,700.0 | 2,183.0 | 3,341.4 |
| `extract` (warm) | 10 | 1,361.3 | 1,182.4 | 1,572.9 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| `summarize` (warm) | 31.0 | 5.8% | 528.5 | 619.8 | -7.0% |
| `classify` (warm) | 354.7 | 13.1% | 2,567.9 | 3,341.4 | +17.9% |
| `extract` (warm) | 107.4 | 7.9% | 1,356.8 | 1,572.9 | -1.7% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, `summarize` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 620 | 521 | 525 | 512 | 541 | 533 | 532 | 512 | 532 | 529 |

Per-segment mean (ms) across the run, `classify` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 2,545 | 2,568 | 2,183 | 2,662 | 2,452 | 2,353 | 2,869 | 3,341 | 3,054 | 2,973 |

Per-segment mean (ms) across the run, `extract` (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 1,356 | 1,357 | 1,420 | 1,381 | 1,573 | 1,404 | 1,271 | 1,182 | 1,257 | 1,411 |

### `bge-small` (embeddings)

Embed benchmark failed: No GGUF files found in repository: BAAI/bge-small-en-v1.5. For llama.cpp, use a GGUF repo ID or a local GGUF file.

### `minilm-l6` (embeddings)

Embed benchmark failed: No GGUF files found in repository: sentence-transformers/all-MiniLM-L6-v2. For llama.cpp, use a GGUF repo ID or a local GGUF file.

### `gemma4-e2b` (VL)

Prompt: 'Describe this receipt in one sentence.' with `receipt1.jpeg`.

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First VL call (includes model load) | 1 | 6,683.6 | 6,683.6 | 6,683.6 |
| VL describe-image (warm) | 10 | 485.3 | 460.9 | 504.4 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| VL describe-image (warm) | 13.2 | 2.7% | 481.6 | 504.4 | -4.0% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, VL describe-image (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 493 | 496 | 461 | 480 | 492 | 504 | 497 | 482 | 476 | 474 |

### `gemma4-e4b` (VL)

Prompt: 'Describe this receipt in one sentence.' with `receipt1.jpeg`.

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First VL call (includes model load) | 1 | 16,663.1 | 16,663.1 | 16,663.1 |
| VL describe-image (warm) | 10 | 645.3 | 625.3 | 702.6 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| VL describe-image (warm) | 25.8 | 4.0% | 629.6 | 702.6 | -3.8% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, VL describe-image (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 643 | 661 | 703 | 673 | 629 | 629 | 630 | 630 | 625 | 630 |

### `qwen3-vl-4b` (VL)

Prompt: 'Describe this receipt in one sentence.' with `receipt1.jpeg`.

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First VL call (includes model load) | 1 | 4,563.8 | 4,563.8 | 4,563.8 |
| VL describe-image (warm) | 10 | 740.1 | 633.1 | 811.0 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| VL describe-image (warm) | 57.2 | 7.7% | 754.2 | 811.0 | -5.6% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, VL describe-image (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 811 | 660 | 786 | 788 | 749 | 761 | 754 | 704 | 755 | 633 |

### `qwen3-vl-8b` (VL)

Prompt: 'Describe this receipt in one sentence.' with `receipt1.jpeg`.

| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |
|---|---|---|---|---|
| First VL call (includes model load) | 1 | 44,096.7 | 44,096.7 | 44,096.7 |
| VL describe-image (warm) | 10 | 960.1 | 913.9 | 1,017.3 |

Consistency:

| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |
|---|---|---|---|---|---|
| VL describe-image (warm) | 36.8 | 3.8% | 946.8 | 1,017.3 | -5.5% |

CV is the coefficient of variation (stdev / mean). Drift compares the mean of the last quarter of calls against the first quarter; positive drift means calls slowed down over the run.

Per-segment mean (ms) across the run, VL describe-image (warm):

| S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 |
|---|---|---|---|---|---|---|---|---|---|
| 947 | 1,002 | 938 | 959 | 1,017 | 998 | 978 | 921 | 914 | 927 |

## Notes

- Structured tasks (`classify`, `extract`) include JSON parsing and
  may retry on validation failures, so their timings can exceed a
  single `complete()` call.
- VL (image) inputs are supported by the llama.cpp runtime for
  Gemma and Qwen VL GGUF models only.
- Only models from the `aibackends.models` catalog are benchmarked.

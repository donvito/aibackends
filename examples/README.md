# Examples

This directory is organized around local runtime examples. The examples run
directly with `llamacpp` or `transformers` in plain Python.

It contains two kinds of examples:

- Task examples that create configured task objects with `create_task(TaskClass, ...)`
- Workflow examples that create configured pipelines with `create_workflow(WorkflowClass, ...)` or explicit pipeline construction

## Setup

From the repo root:

```bash
python3 -m pip install -e ".[dev]"
```

Choose and install a runtime:

```bash
# llama.cpp
python3 -m pip install -e ".[llamacpp-metal]"

# Transformers
python3 -m pip install -e ".[transformers]"
```

Task examples use `create_task(TaskClass, ...)` with supported runtime/model
refs such as `LLAMACPP` and `GEMMA4_E2B`, so defaults are configured before
`run(...)`.

`basic_task_transformers.py` uses the smaller `GEMMA3_270M_IT` profile so it
stays practical on CPU-only machines. If you swap it to `GEMMA4_E2B`, expect a
much larger first download and slower load time.

`embed_text_transformers.py` and `workflows/embedding_similarity.py` use
`MINILM_L6`, a compact local embeddings profile that stays practical on
CPU-only machines.

`redact_text.py` and `redact_text_batch.py` use local PII backends rather than
the general `llamacpp` or `transformers` runtimes.

`redact_text_gliner_cache_benchmark.py` demonstrates the difference between the
first cold GLiNER load and repeated warm calls in the same Python process using
a larger repeated contract sample.

`redact_text_gliner_cuda_benchmark.py` runs the same repeated-call benchmark
with `device="cuda"` so you can compare CPU and CUDA timings on that larger
input.

`redact_text_gliner_100_call_benchmark.py` runs the larger CPU benchmark 100
times and compares the total time against the committed `main` baseline to make
the long-run trade-off easier to judge.

`workflows/image_ocr_gemma.py` and `workflows/image_ocr_qwen.py` are vision
OCR examples that extract structured receipt JSON from the sample receipt
images in `examples/data/images/` using the `llamacpp` runtime.

`workflows/image_understanding_gemma.py` and
`workflows/image_understanding_qwen.py` are lighter image-understanding demos
that return a short description plus any visible text. The Qwen examples expect
a recent vision-capable `llama-cpp-python` build with Qwen VL handler support.

## Runnable core examples

These examples use the sample files in `examples/data/`.
They are the best starting point if you want to run models locally.

```bash
python3 examples/list_available.py
python3 examples/tasks/basic_task.py
python3 examples/tasks/basic_task_transformers.py
python3 examples/tasks/embed_text_transformers.py
python3 examples/tasks/summarize_text.py
python3 examples/tasks/classify_text.py
python3 examples/tasks/redact_text.py
python3 examples/tasks/redact_text_gliner_cache_benchmark.py
python3 examples/tasks/redact_text_gliner_cuda_benchmark.py
python3 examples/tasks/redact_text_gliner_100_call_benchmark.py
python3 examples/tasks/redact_text_batch.py
python3 examples/tasks/extract_custom_schema.py
python3 examples/tasks/task_interface.py
python3 examples/tasks/sales_call_report.py
python3 examples/tasks/video_ad_report.py
python3 examples/workflows/audio_transcribe.py
python3 examples/workflows/batch_processing.py
python3 examples/workflows/custom_pipeline.py
python3 examples/workflows/embedding_similarity.py
python3 examples/workflows/image_ocr_gemma.py
python3 examples/workflows/image_ocr_qwen.py
python3 examples/workflows/image_understanding_gemma.py
python3 examples/workflows/image_understanding_qwen.py
python3 examples/workflows/invoice_redact_extract.py
python3 examples/workflows/resume_redact_summarize.py
python3 examples/workflows/resume_role_match.py
python3 examples/workflows/support_transcript_redact_validate.py
```

`list_available.py` has no runtime dependency. It prints the supported runtime
and model catalog plus the canonical task/workflow names returned by the public
`available_*()` helpers.

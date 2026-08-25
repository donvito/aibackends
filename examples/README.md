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

# GliGuard prompt/response moderation
python3 -m pip install -e ".[guardrails]"

# GLiNER 2.5 information extraction
python3 -m pip install -e ".[information-extraction]"
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

`tool_calling_lfm.py` demos native tool calling with `LFM25_2_6B`
(LiquidAI LFM2.5-2.6B). It works on both runtimes and exposes CPU/GPU and
GGUF quantization toggles:

```bash
python3 examples/tasks/tool_calling_lfm.py --runtime llamacpp --device cpu
python3 examples/tasks/tool_calling_lfm.py --runtime transformers --device cpu
python3 examples/tasks/tool_calling_lfm.py --runtime llamacpp --quantization Q8_0
```

`redact_text.py` and `redact_text_batch.py` use local PII backends rather than
the general `llamacpp` or `transformers` runtimes.

`moderate_content.py` demonstrates GliGuard prompt safety, toxicity, jailbreak
detection, response safety, refusal/compliance detection, and native batch
inference. Select CPU or GPU explicitly:

```bash
python3 examples/tasks/moderate_content.py --device cpu
python3 examples/tasks/moderate_content.py --device gpu
```

`gliner25_information_extraction.py` covers the boundary architecture's main
use cases: long-document contract review with global offsets, schema-shaped
invoice records, span-level sentiment, constrained agent routing, typed
knowledge graphs, and multilingual entities. The default 74M model is practical
on CPU; choose the 194M English model or 287M multilingual model explicitly:

```bash
python3 examples/tasks/gliner25_information_extraction.py --model small
python3 examples/tasks/gliner25_information_extraction.py --model base
python3 examples/tasks/gliner25_information_extraction.py \
    --model multi --use-case multilingual
```

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
python3 examples/tasks/moderate_content.py --device cpu
python3 examples/tasks/gliner25_information_extraction.py --model small
python3 examples/tasks/redact_text.py
python3 examples/tasks/redact_text_batch.py
python3 examples/tasks/extract_custom_schema.py
python3 examples/tasks/task_interface.py
python3 examples/tasks/tool_calling_lfm.py
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

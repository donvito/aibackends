# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LFM2.5 prompt routing backend powered by
  `LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`, a 350M bidirectional encoder
  that scores a prompt against free-text routing lanes in one forward pass.
  It runs as its own local capability backend through `transformers`
  (`trust_remote_code=True`), not through the configured generative runtime,
  with CPU/CUDA/MPS device selection and per-process, per-device model reuse.
- Routing task `route_prompt`, batch form `route_prompts`, and `_async`
  variants of both, exported from `aibackends.tasks` and the top-level
  `aibackends` package, plus `RoutePromptTask` for the `create_task(...)` API.
- `RoutingResult` and `RouteScore` schemas in `aibackends.schemas.routing`
  with ranked scores, a `best_route` convenience field, and threshold
  filtering (below-threshold lanes are dropped; `best_route` is `None` when
  nothing clears the bar).
- Pluggable routing backend registry (`register_routing_backend`,
  `get_routing_backend`, `list_routing_backends`) under
  `aibackends.backends.routing`.
- New `routing` extra (`pip install aibackends[routing]`) pulling in `torch`
  and `transformers`.
- CLI task `route-prompt`; routing lanes ride the existing `--labels` flag
  and `--threshold` / `--device` apply.
- `qwen3.8-27b` llamacpp model profile (`unsloth/Qwen3.8-27B-GGUF`, `Q4_K_M`)
  and `QWEN38_27B` ref as a local dispatch target for routing demos.
- Runnable examples under `examples/routing/`, one file per scenario: basic
  routing, device-assistant orchestration, code-language routing,
  support-intent classification, on-the-fly custom categories, capability
  dispatch to GliGuard/GLiNER, and complexity-based model-tier routing
  across local and cloud models — plus a Colab notebook at
  `examples/notebooks/lfm25_prompt_routing_colab.ipynb`.

### Fixed
- The llama.cpp runtime now strips transformers-only `{% generation %}`
  markers from GGUF-embedded chat templates before llama-cpp-python compiles
  them. Recent LiquidAI LFM2.5 GGUF refreshes ship such templates, which
  previously crashed `Llama(...)` construction with a Jinja
  `TemplateSyntaxError`.
- The llama.cpp runtime honours `extra_options={"chat_format": ...}` for all
  models, not only Gemma, as an escape hatch for incompatible embedded
  templates.
- The routing backend falls back to the fast tokenizer when the router
  repo's `tokenizer_config.json` names a transformers v5 tokenizer class
  that transformers 4.x does not recognize, so it works on both major
  versions (the 4.x ceiling is currently imposed by `gliner`/`gliner2`).

## [0.6.0] - 2026-08-25

### Added
- GLiNER2.5 (`fastino/gliner2.5-*-v1`) extraction backend running as its own
  local capability backend, not through the configured generative runtime.
  Three variants are selectable by name: `small` (74M), `base` (194M,
  default), and `multi` (287M, multilingual), or any Hugging Face repo id.
- Entity extraction with `extract_entities` / `extract_entities_batch`:
  zero-shot labels, character offsets and confidences, span attributes
  (qualified spans), unlimited span lengths, and long-document mode with
  configurable chunk size and overlap.
- Constrained classification with `classify_text` / `classify_texts`:
  multi-task label sets, multi-label mode, per-label probabilities, and
  `implies` / `excludes` / `iff` constraints with a `feasible` flag.
- Joint entity-relation extraction with `extract_graph`, returning a typed
  `KnowledgeGraph` of entities and typed relations with optional
  `no_self_loops`.
- `_async` variants of every extraction task, plus `ExtractEntitiesTask`,
  `ClassifyTextTask`, and `ExtractGraphTask` for the `create_task(...)` API.
  All extraction tasks are also exported from the top-level `aibackends`
  package.
- `EntityExtraction`, `TextClassification`, and `KnowledgeGraph` schemas in
  `aibackends.schemas.extraction`, with `ExtractedEntity`, `SpanAttribute`,
  `TaskClassification`, `GraphEntity`, and `GraphRelation`.
- Pluggable extraction backend registry (`register_extraction_backend`,
  `get_extraction_backend`, `list_extraction_backends`) under
  `aibackends.backends.extraction`.
- New `extraction` extra (`pip install aibackends[extraction]`) pulling in
  `gliner2[local]` and `protobuf`.
- CLI tasks `extract-entities`, `classify-text`, and `extract-graph`, with new
  `--entities` and repeatable `--relation name:head:tail` flags.
- Seven runnable examples under `examples/gliner25/` (PII offsets, unlimited
  spans, knowledge graph, constrained routing, span attributes, long
  documents, multilingual NER) and an executed Colab notebook at
  `examples/notebooks/gliner25_extraction_colab.ipynb`.
- GLiNER2.5 CPU benchmark covering cold load, warm latency per task type,
  native batch throughput, and long-document scaling across all three
  variants, plus a committed report in `benchmarks/reports/`.
- GLiNER2.5 zero-shot accuracy eval over ag_news, rotten_tomatoes, and
  CrossNER politics, with a committed report in `evals/reports/`.
- Capability backend architecture documented across the PII, moderation, and
  extraction backends in `docs/architecture.md`, `docs/concepts.md`, and
  `docs/extending.md`.

## [0.5.0] - 2026-08-25

### Added
- GliGuard (`fastino/gliguard-LLMGuardrails-300M`) moderation backend with
  typed prompt and response tasks, all six documented safety signals, native
  batch inference, model reuse, and CPU/CUDA/MPS device selection.
- Moderation tasks `moderate_prompt`, `moderate_response`, their batch forms
  `moderate_prompts` / `moderate_responses`, and `_async` variants of all four,
  exported from the top-level `aibackends` package.
- `PromptModeration` and `ResponseModeration` schemas in
  `aibackends.schemas.moderation`, with `SafetyVerdict`, `RefusalVerdict`,
  `HarmCategory`, and `JailbreakStrategy` label types.
- Pluggable moderation backend registry
  (`register_moderation_backend`, `get_moderation_backend`,
  `list_moderation_backends`) under `aibackends.backends.moderation`.
- New `guardrails` extra (`pip install aibackends[guardrails]`) pulling in
  `gliner2[local]` and `protobuf`; both are also part of the `all` extra.
- CLI flags `--prompt`, `--device`, `--threshold`, and `--category-threshold`
  on `aibackends run`, applied only to tasks that accept them.
- GliGuard CPU benchmark covering process-cold load/first-call cost, warm
  prompt and response latency, and native batch throughput, plus a committed
  report in `benchmarks/reports/`.
- Moderation example `examples/tasks/moderate_content.py`.

### Changed
- The CLI `--backend` flag no longer defaults to `gliner`; when it is omitted,
  each task now picks its own default backend.

## [0.4.0] - 2026-08-16

### Added
- LiquidAI LFM2.5-2.6B support via the new `LFM25_2_6B` model ref:
  `LiquidAI/LFM2.5-2.6B` on `transformers` and `LiquidAI/LFM2.5-2.6B-GGUF`
  on `llamacpp`, with Liquid AI's recommended generation defaults
  (`temperature=0.1`, `top_k=50`, `repetition_penalty=1.1`, `bfloat16` on
  `transformers`).
- Configurable GGUF quantization: new `quantization` config field and
  per-model profile default (LFM2.5 defaults to `Q4_K_M`); falls back to the
  hardware default when unset.
- CPU/GPU toggle via the `device` config field on both local runtimes:
  `llamacpp` maps `"cpu"`/`"gpu"` to GPU layer offload and `transformers`
  maps it to `device_map` (including `"gpu"` -> `"cuda"`).
- Both runtimes now forward `top_k`, `top_p`, `repetition_penalty` (and
  `min_p` for `llamacpp`) from per-call kwargs, `extra_options`, or model
  profile generation defaults. `transformers` also honours a `dtype` load
  option and a `skip_special_tokens` decode option.
- Tool-calling example `examples/tasks/tool_calling_lfm.py` demoing LFM2.5's
  native Pythonic tool-call format on either runtime with CPU/GPU and
  quantization flags.
- `aibackends.core.tool_calls` with `ToolCall`, `extract_tool_calls`,
  `strip_reasoning`, and `clean_answer` for parsing Pythonic tool calls from
  model responses (with or without `<|tool_call_start|>` markers).
- Tool-call accuracy eval `evals/eval_tool_calls.py` scoring tool selection,
  argument accuracy, and exact match over a labeled case set (single-tool,
  multi-tool, and no-tool questions), with dated reports in `evals/reports/`.
- CPU benchmark reports for LFM2.5-2.6B (runtime reuse and task latency on
  `llamacpp` and `transformers`) in `benchmarks/reports/`.
- LiquidAI LFM2.5-VL-3B vision support via the new `LFM25_VL_3B` model ref:
  `LiquidAI/LFM2.5-VL-3B-GGUF` on `llamacpp` (default `Q4_K_M`), with a
  ChatML multimodal chat handler and automatic `mmproj` projector download.
  Receipt-extraction demo in `examples/workflows/image_ocr_lfm.py`.
- `--device` flag on `benchmarks/benchmark_tasks.py` to force CPU or GPU
  inference, plus a committed Q4_K_M CPU report for the LFM2.5-VL-3B VL task
  in `benchmarks/reports/`.

### Fixed
- `parse_json_content` now ignores JSON drafted inside a reasoning
  (`<think>...</think>`) block, so structured tasks work with reasoning
  models such as LFM2.5 on runtimes without grammar-constrained output.

## [0.3.0] - 2026-07-18

### Added
- Runtime reuse: `get_runtime(...)` now caches runtime instances process-wide
  (keyed by the load-affecting config fields), so the model loaded on the
  first task call stays warm and subsequent calls skip the load entirely.
  Controlled by the new `reuse_runtime` config flag (default `true`); set
  `configure(reuse_runtime=False)` or pass `reuse_runtime=False` per call to
  restore the previous build-per-call behavior.
- `aibackends.preload(...)` and `BaseRuntime.preload()` to load a runtime's
  model ahead of the first request, mirroring the PII `backend.load()`
  pre-warm API.
- `aibackends.clear_runtime_cache()` to drop cached runtime instances and
  release the RAM/VRAM held by loaded models. `reset_config()` clears the
  cache as well.
- The `openai-privacy` PII backend now builds its token-classification
  pipeline once per process and caches it under a thread-safe lock (mirroring
  the GLiNER cache), and sets `load_model` so `backend.load()` pre-warms it.
  Exposes `load_privacy_pipeline` and `clear_pipeline_cache`.
- Benchmark suite under `benchmarks/` (runtime reuse, per-model task latency
  including VL image tasks, PII backends) with dated markdown reports
  committed to `benchmarks/reports/`, consistency/drift statistics, a
  sequential `run_all.py` driver, and a lock file so benchmarks never run
  concurrently on a single GPU.

### Changed
- `LlamaCppRuntime` and `TransformersRuntime` serialize model loading and
  inference behind a per-instance lock, since reused instances can now be
  shared across threads.

## [0.2.1] - 2026-04-29

### Changed
- Trim source distribution: future sdists no longer ship `.vscode/`,
  `.pre-commit-config.yaml`, `uv.lock`, `markdown-preview.css`, or the
  binary example assets under `examples/data/{audio,images,pdf}/`.
  Drops the sdist from ~3 MB to ~77 KB. The wheel is unchanged.

## [0.2.0] - 2026-04-29

### Added
- `PIIBackendSpec.load()` and `PIIBackendSpec.redact(text, *, labels=...)` so
  callers can drive a PII backend natively, e.g.
  `get_pii_backend("gliner").redact(text, labels=PII_LABELS)`.
- New `load_model` field on `PIIBackendSpec` for backends that own an
  in-process model handle.
- Shared `aibackends.backends.pii.apply_redactions(text, entities,
  backend_name=...)` helper used by both the spec method and the
  `redact_pii` task.
- GLiNER backend now loads `gliner.GLiNER.from_pretrained(...)` once per
  process and caches it under a thread-safe lock, so batch redaction skips
  the heavy load cost after the first call. Exposes `load_gliner_model`
  and `clear_model_cache` for tests and explicit pre-warming.

### Changed
- `aibackends.tasks.redact_pii` is now a thin wrapper that delegates to
  `backend_spec.redact(...)`.
- `examples/tasks/redact_text_batch.py` simplified to use the new native
  API (`backend.load()` + `backend.redact(...)`).
- `docs/extending.md` updated to describe `backend.load()` /
  `backend.redact(...)` ergonomics for new PII backends.

### Removed
- The subprocess-based `aibackends.backends.pii.gliner.worker` module and
  its `GLINER_WORKER_PATH` constant. GLiNER now runs in-process; any
  external code that imported `GLINER_WORKER_PATH` should switch to
  `backend.load()` / `backend.redact(...)`.

## [0.1.0] - 2026-04-25

### Added
- Initial public release.

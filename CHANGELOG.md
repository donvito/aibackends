# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

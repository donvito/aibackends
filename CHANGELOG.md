# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

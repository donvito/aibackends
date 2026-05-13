# AI Agent Guidance

## Python checks

This repo uses Ruff and Mypy settings from `pyproject.toml`. When editing Python:

- Keep every Python line at `100` characters or fewer. Wrap long `print(...)`, f-strings, imports, and function calls before finishing.
- Keep imports clean and grouped so Ruff passes (`B`, `E`, `F`, `I`, `UP` are enabled).
- For code in `src/` and `tests/`, keep Mypy green:
  - Add explicit parameter and return annotations to new or changed functions.
  - Do not rely on implicit optional types; write `T | None` explicitly.
  - Remember `check_untyped_defs = true`, so even partially annotated helpers must still type-check.
  - Prefer specific return types over `Any` unless there is a strong reason not to.
- `examples/` is not in the current Mypy target list, but example files must still pass Ruff and should stay simply typed.

## Before finishing Python work

1. Run `ReadLints` on edited files.
2. Run `python -m ruff check` on edited Python files when the change is non-trivial.
3. If `src/` or `tests/` changed, run `python -m mypy`.
4. For edited runnable examples, use `python -m compileall` or run the example directly when practical.

## Cursor Cloud specific instructions

- **No external services required.** This is a pure-Python library with no databases, Docker, or API servers to start. All tests use a built-in `StubRuntime` (see `tests/conftest.py`).
- **Dev commands** (see `README.md § Development` and `CONTRIBUTING.md`):
  - Lint: `python3 -m ruff check .`
  - Type-check: `python3 -m mypy src tests`
  - Tests: `python3 -m pytest tests`
- **CLI entry point** is installed to `~/.local/bin/aibackends`. Ensure `$HOME/.local/bin` is on `PATH` before invoking `aibackends` directly, or use `python3 -m aibackends.cli` instead.
- **Mypy known caveat:** `python3 -m mypy src tests` may report one pre-existing error in `src/aibackends/core/runtimes/llamacpp.py` about the optional `llama_cpp.llama_chat_format` sub-module. This is expected when `llama-cpp-python` is not installed (it is an optional extra) and does not indicate a code defect.

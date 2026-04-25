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

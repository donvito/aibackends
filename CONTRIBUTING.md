# Contributing

## Development setup

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest tests
python3 -m mypy src tests
ruff check .
```

## Adding a new task

1. Create one file in `src/aibackends/tasks/`.
2. Add or update the output schema in `src/aibackends/schemas/`.
3. Re-export the task in `src/aibackends/tasks/__init__.py`.
4. Add focused coverage in `tests/tasks/`.

## Adding a new workflow

1. Compose steps in `src/aibackends/workflows/`.
2. Reuse existing step primitives where possible.
3. Add batch and error-path coverage in `tests/workflows/`.

## Adding a new runtime

1. Implement the `BaseRuntime` contract in `src/aibackends/core/runtimes/`.
2. Register it in `src/aibackends/core/config.py`.
3. Add mocked tests in `tests/runtimes/`.

## Code style

- Ruff for linting
- Type hints required
- Google-style docstrings for public functions when extra context is helpful

## Testing

- Use `pytest`
- Prefer mocked or stubbed runtimes in CI
- Keep tests focused on API behavior, config merging, validation, and orchestration

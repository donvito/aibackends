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
2. Implement `BaseTask.run(...)`.
3. Add or update the output schema in `src/aibackends/schemas/`.
4. Export `TASK_SPEC` from the task module.
5. Add focused coverage in `tests/tasks/`.

## Adding a new workflow

1. Compose steps in `src/aibackends/workflows/`.
2. Export `WORKFLOW_SPEC` from the workflow module.
3. Use `WorkflowSpec(..., workflow_factory=YourWorkflow)`.
4. Reuse existing step primitives where possible.
5. Add batch and error-path coverage in `tests/workflows/`.

## Adding a new runtime

1. Implement the `BaseRuntime` contract in `src/aibackends/core/runtimes/`.
2. Export `RUNTIME_SPEC` from the runtime module.
3. Add mocked tests in `tests/runtimes/`.

## Adding a model profile

1. Add a file under `src/aibackends/models/`.
2. Export `MODEL_PROFILE` or `MODEL_PROFILES`.
3. Use profiles for aliases, chat templates, and model-specific defaults.

## Adding a capability backend

Use a backend for a swappable implementation of one capability, not for a
general LLM runtime. PII backends live under `src/aibackends/backends/pii/`.

1. Add a file or package for the backend.
2. Export the backend spec, such as `PII_BACKEND_SPEC`.
3. Keep backend-owned helpers inside that backend package.

## Code style

- Ruff for linting
- Type hints required
- Google-style docstrings for public functions when extra context is helpful

## Testing

- Use `pytest`
- Prefer mocked or stubbed runtimes in CI
- Keep tests focused on API behavior, config merging, validation, and orchestration

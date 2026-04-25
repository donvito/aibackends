# API Reference

The public API is grouped into:

- `aibackends.configure(...)`
- `aibackends.tasks.*`
- `aibackends.tasks.BaseTask`
- `aibackends.tasks.create_task(...)`
- `aibackends.workflows.*`
- `aibackends.workflows.create_workflow(...)`
- `aibackends.core.runtimes.*`
- `aibackends.core.registry.TaskSpec`
- `aibackends.core.registry.WorkflowSpec`
- `aibackends.tasks.get_task(...)`
- `aibackends.tasks.register_task(...)`
- `aibackends.workflows.get_workflow(...)`

The codebase is structured to keep task functions framework-agnostic while
allowing runtime, backend, model profile, and workflow internals to evolve
independently.
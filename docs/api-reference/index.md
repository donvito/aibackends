# API Reference

The public API is grouped into:

- `aibackends.configure(...)`
- `aibackends.available_runtimes()`
- `aibackends.available_models()`
- `aibackends.runtimes.*`
- `aibackends.models.*`
- `aibackends.runtimes.get_runtime_spec(name)`
- `aibackends.models.get_model_ref(name)`
- `aibackends.tasks.*`
- `aibackends.tasks.moderate_prompt(...)`
- `aibackends.tasks.moderate_response(...)`
- `aibackends.tasks.extract_entities(...)`
- `aibackends.tasks.route_agent(...)`
- `aibackends.tasks.screen_agent_action(...)`
- `aibackends.tasks.extract_memory_graph(...)`
- `aibackends.tasks.review_contract(...)`
- `aibackends.tasks.extract_clinical(...)`
- `aibackends.backends.moderation.*`
- `aibackends.backends.extraction.*`
- `aibackends.tasks.BaseTask`
- `aibackends.tasks.create_task(TaskClass, ...)`
- `aibackends.workflows.*`
- `aibackends.workflows.create_workflow(WorkflowClass, ...)`
- `aibackends.core.runtimes.*`
- `aibackends.core.registry.TaskSpec`
- `aibackends.core.registry.WorkflowSpec`
- `aibackends.tasks.get_task(name)`
- `aibackends.tasks.register_task(...)`
- `aibackends.workflows.get_workflow(name)`

The codebase is structured to keep task functions framework-agnostic while
allowing runtime, backend, model profile, and workflow internals to evolve
independently.
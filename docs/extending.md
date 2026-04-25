# Extending

AIBackends is designed so built-in extensions are visible as files. Add the
file, export the spec, and add focused tests.

## Add A Runtime

Use this for a new general LLM/embedding executor.

Create a module under `src/aibackends/core/runtimes` and export `RUNTIME_SPEC`:

```python
from aibackends.core.registry import RuntimeSpec
from aibackends.core.runtimes.base import OpenAICompatibleRuntime


class MyRuntime(OpenAICompatibleRuntime):
    provider_name = "my-runtime"
    default_base_url = "http://localhost:8000/v1"


RUNTIME_SPEC = RuntimeSpec(name="my-runtime", factory=MyRuntime)
```

## Add A Transformer Model Profile

Use this when model support needs model-specific configuration such as aliases or
chat templates.

Create a module under `src/aibackends/models`:

```python
from aibackends.core.registry import TransformerModelProfile


MODEL_PROFILE = TransformerModelProfile(
    name="my-instruct-model",
    model_id="org/my-instruct-model",
    chat_template="{{ messages }}",
    runtime="transformers",
)
```

Explicit user config such as `chat_template` or `chat_template_path` takes
precedence over profile defaults.

## Add A Capability Backend

Use this for a swappable implementation behind one capability, not for a general
LLM runtime.

PII backends currently live under `src/aibackends/backends/pii`. The GLiNER
backend is a package because it owns an implementation helper:

```text
src/aibackends/backends/pii/gliner/
  __init__.py
  worker.py
```

The backend package exports `PII_BACKEND_SPEC`.

## Add A Task

Create one task module under `src/aibackends/tasks`, implement `BaseTask`, and
export `TASK_SPEC`:

```python
from aibackends.core.registry import ModelRefLike, RuntimeRefLike, TaskSpec
from aibackends.tasks import BaseTask


class MyTask(BaseTask):
    name = "my-task"

    def run(
        self,
        input: str,
        *,
        runtime: RuntimeRefLike | None = None,
        model: ModelRefLike | None = None,
    ):
        ...


TASK_SPEC = TaskSpec(name=MyTask.name, task_factory=MyTask)
```

Add or update the output schema in `src/aibackends/schemas` when the task returns
structured output. Public function wrappers are still useful for the simple
`from aibackends.tasks import my_task` API.

Task instances can be configured before they run:

```python
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import create_task

task = create_task(MyTask, runtime=LLAMACPP, model=GEMMA4_E2B)
result = task.run("input text")
```

## Add A Workflow

Create one workflow module under `src/aibackends/workflows`, compose existing
steps where possible, and export `WORKFLOW_SPEC`:

```python
from aibackends.core.registry import WorkflowSpec
from aibackends.workflows import Pipeline


class MyWorkflow(Pipeline):
    steps = [...]


WORKFLOW_SPEC = WorkflowSpec(name="my-workflow", workflow_factory=MyWorkflow)
```

Workflow steps receive a `StepContext` object with the workflow name and resolved
runtime config.

Workflow instances can be configured before they run:

```python
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.workflows import create_workflow

workflow = create_workflow(MyWorkflow, runtime=LLAMACPP, model=GEMMA4_E2B)
result = workflow.run("input.txt")
```

## Test Expectations

- Runtime tests should use mocked providers or the stub runtime.
- Model profile tests should verify alias resolution and config precedence.
- Backend tests should verify backend-specific defaults and errors.
- Task tests should focus on API behavior and validation.
- Workflow tests should cover orchestration and batch behavior.


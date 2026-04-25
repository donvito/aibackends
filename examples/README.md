# Examples

This directory contains three kinds of examples:

- Task examples that create configured task objects with `create_task(TaskClass, ...)`
- Workflow examples that create configured pipelines with `create_workflow(WorkflowClass, ...)` or explicit pipeline construction
- Agent framework integrations that wrap configured task objects as tools

The agent examples intentionally keep AIBackends tasks and workflows outside the
agent framework. The framework layer only wraps `task.run(...)` or
`workflow.run(...)`, so the same pipeline can be reused if you switch frameworks
or run more than one framework in the same application.

## Setup

From the repo root:

```bash
python3 -m pip install -e ".[dev]"
```

Choose and install a runtime:

```bash
# llama.cpp
python3 -m pip install -e ".[llamacpp-metal]"

# Transformers
python3 -m pip install -e ".[transformers]"
```

Task examples use `create_task(TaskClass, ...)` with supported runtime/model
refs such as `LLAMACPP` and `GEMMA4_E2B`, so defaults are configured before
`run(...)`. Agent examples follow the same pattern: create the task object once,
then expose `task.run(...)` through the framework's tool wrapper.

`basic_task_transformers.py` uses the smaller `GEMMA3_270M_IT` profile so it
stays practical on CPU-only machines. If you swap it to `GEMMA4_E2B`, expect a
much larger first download and slower load time.

## Runnable core examples

These examples use the sample files in `examples/data/` and do not require any agent framework package.

```bash
python3 examples/list_available.py
python3 examples/tasks/basic_task.py
python3 examples/tasks/basic_task_transformers.py
python3 examples/tasks/summarize_text.py
python3 examples/tasks/classify_text.py
python3 examples/tasks/redact_text.py
python3 examples/tasks/extract_custom_schema.py
python3 examples/tasks/task_interface.py
python3 examples/tasks/sales_call_report.py
python3 examples/tasks/video_ad_report.py
python3 examples/workflows/batch_processing.py
python3 examples/workflows/custom_pipeline.py
python3 examples/workflows/resume_redact_summarize.py
python3 examples/workflows/resume_role_match.py
```

## Framework integration examples

These examples require the corresponding framework packages to be installed separately:

- `examples/agents/langgraph_agent.py`
- `examples/agents/pydantic_ai_agent.py`
- `examples/agents/openai_agents_sdk.py`
- `examples/agents/crewai_agent.py`
- `examples/agents/agno_agent.py`
- `examples/agents/llamaindex_agent.py`

They are intended to show how AIBackends tasks plug into agent frameworks as tools.

`list_available.py` has no runtime dependency. It prints the supported runtime
and model catalog plus the canonical task/workflow names returned by the public
`available_*()` helpers.

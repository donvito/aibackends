# Examples

This directory contains two kinds of examples:

- Runnable core-task examples that use bundled sample text files in `examples/data/`
- Agent framework integrations that show how to wrap AIBackends tasks as tools

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

Examples that call `configure(...)` can swap runtimes by changing that call. `redact_text.py` is different: it uses `redact_pii(..., backend="gliner")` and does not depend on the global runtime configuration.

`basic_task_transformers.py` uses the smaller instruction-tuned `google/gemma-3-270m-it` model so it stays practical on CPU-only machines. If you swap it to `gemma4-e2b`, expect a much larger first download and slower load time.

## Runnable core examples

These examples use the sample files in `examples/data/` and do not require any agent framework package.

```bash
python3 examples/basic_task.py
python3 examples/basic_task_transformers.py
python3 examples/summarize_text.py
python3 examples/classify_text.py
python3 examples/redact_text.py
python3 examples/extract_custom_schema.py
python3 examples/sales_call_report.py
python3 examples/video_ad_report.py
python3 examples/batch_processing.py
python3 examples/custom_pipeline.py
```

## Framework integration examples

These examples require the corresponding framework packages to be installed separately:

- `langgraph_agent.py`
- `pydantic_ai_agent.py`
- `openai_agents_sdk.py`
- `crewai_agent.py`
- `agno_agent.py`
- `llamaindex_agent.py`

They are intended to show how AIBackends tasks plug into agent frameworks as tools.

# Agent Integration

The integration strategy is simple: AIBackends tasks are just Python callables.

## LangGraph

Wrap the task with `@tool` and pass it into your agent.

## pydantic-ai

Pass the task directly in `tools=[...]`.

## OpenAI Agents SDK

Wrap the task with `@function_tool`.

## CrewAI

Wrap the task with `@tool("Name")`.

## Agno

Wrap the task with `@tool` and register it on the agent.

## LlamaIndex

Use `FunctionTool.from_defaults(fn=...)`.

See the `examples/` directory for one example per framework.

# AIBackends

AIBackends is a Python library of ready-made AI tasks that plug into any agent framework as tools.

It is built around three ideas:

- Tasks first: one function call in, one typed result out.
- Local-first runtimes: `llama-cpp-python` and `transformers` are first-class.
- Framework agnostic: LangGraph, pydantic-ai, OpenAI Agents SDK, CrewAI, Agno, LlamaIndex, or your own code.

```python
from aibackends import configure
from aibackends.tasks import extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")

result = extract_invoice("invoice.pdf")
print(result.total)
```

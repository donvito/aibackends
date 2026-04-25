# Runtimes

AIBackends separates tasks from inference runtimes.

## First-class local runtimes

- `llamacpp`: in-process GGUF inference with automatic model resolution and caching
- `transformers`: in-process Hugging Face models, optional adapters, optional 4-bit loading

## Supported local server runtimes

- `ollama`
- `lmstudio`

## Supported cloud runtimes

- `anthropic`
- `together`
- `groq`

## Configure a default

```python
from aibackends import configure

configure(runtime="llamacpp", model="gemma4-e2b")
```

## Per-task override

```python
from aibackends.tasks import summarize

text = summarize("notes.txt", runtime="anthropic", model="claude-sonnet-4-5")
```

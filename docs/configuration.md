# Configuration

Use `configure()` to set global defaults:

```python
from aibackends import configure

configure(
    runtime="transformers",
    model="donvito/viralsense-gemma-270m",
    adapter="donvito/viralsense-lora-v1",
    device="auto",
    load_in_4bit=True,
)
```

You can also load YAML:

```python
from aibackends import load_config

load_config("aibackends.yml")
```

Supported top-level fields map to `RuntimeConfig`.

For local `transformers` models, prompt rendering is configurable:

```python
from aibackends import configure

configure(
    runtime="transformers",
    model="google/gemma-3-270m-it",
    prompt_format="auto",  # auto | chat_template | text
    # chat_template="...",
    # chat_template_path="template.jinja",
)
```

- `prompt_format="auto"` prefers a configured template override, then the tokenizer's own chat template, then falls back to plain text.
- `prompt_format="chat_template"` requires either a tokenizer chat template or an explicit override.
- `prompt_format="text"` always renders plain role-prefixed text.

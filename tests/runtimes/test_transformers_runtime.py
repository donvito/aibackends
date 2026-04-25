from __future__ import annotations

from aibackends.core.prompting import (
    merge_system_messages,
    render_messages_as_text,
    schema_prompt,
)
from aibackends.core.runtimes.transformers import TransformersRuntime
from aibackends.core.types import RuntimeConfig
from aibackends.schemas.invoice import InvoiceOutput


class FakeTensor:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    @property
    def shape(self) -> tuple[int, int]:
        return (1, len(self.values))

    def to(self, device: object) -> FakeTensor:
        del device
        return self


class ChatTemplateTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ built_in_template }}"
        self.pad_token_id = 7
        self.eos_token_id = 9
        self.rendered_prompt: str | None = None
        self.chat_messages: list[dict[str, str]] | None = None
        self.add_generation_prompt: bool | None = None
        self.template_override: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
    ) -> str:
        assert tokenize is False
        self.chat_messages = messages
        self.add_generation_prompt = add_generation_prompt
        self.template_override = chat_template
        return "CHAT TEMPLATE PROMPT"

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, FakeTensor]:
        assert return_tensors == "pt"
        self.rendered_prompt = prompt
        return {
            "input_ids": FakeTensor([101, 102]),
            "attention_mask": FakeTensor([1, 1]),
        }

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        assert tokens == [103]
        return "decoded content"


class PlainTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 3
        self.eos_token_id = None
        self.rendered_prompt: str | None = None

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, FakeTensor]:
        assert return_tensors == "pt"
        self.rendered_prompt = prompt
        return {
            "input_ids": FakeTensor([11, 12]),
            "attention_mask": FakeTensor([1, 1]),
        }

    def decode(self, tokens: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        assert tokens == [103]
        return "plain decoded"


class FakeModel:
    device = None

    def __init__(self) -> None:
        self.eval_called = False
        self.generate_kwargs: dict[str, object] | None = None

    def eval(self) -> FakeModel:
        self.eval_called = True
        return self

    def generate(self, **kwargs: object) -> list[list[int]]:
        self.generate_kwargs = kwargs
        return [[101, 102, 103]]


def test_transformers_runtime_uses_chat_template_when_available():
    runtime = TransformersRuntime(RuntimeConfig(runtime="transformers", model="demo"))
    tokenizer = ChatTemplateTokenizer()
    model = FakeModel()
    runtime._load_generator = lambda: (tokenizer, model)  # type: ignore[method-assign]

    response = runtime.complete(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Return JSON."},
        ]
    )

    assert tokenizer.chat_messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Return JSON."},
    ]
    assert tokenizer.add_generation_prompt is True
    assert tokenizer.rendered_prompt == "CHAT TEMPLATE PROMPT"
    assert model.generate_kwargs is not None
    assert model.generate_kwargs["pad_token_id"] == tokenizer.pad_token_id
    assert model.generate_kwargs["eos_token_id"] == tokenizer.eos_token_id
    assert response.content == "decoded content"
    assert response.raw["prompt_format"] == "chat_template"
    assert response.raw["template_source"] == "tokenizer"


def test_transformers_runtime_merges_leading_system_messages_for_chat_templates():
    runtime = TransformersRuntime(RuntimeConfig(runtime="transformers", model="demo"))
    tokenizer = ChatTemplateTokenizer()
    model = FakeModel()
    runtime._load_generator = lambda: (tokenizer, model)  # type: ignore[method-assign]
    messages = [
        {"role": "system", "content": "Return valid JSON only."},
        {"role": "system", "content": "You extract structured fields."},
        {"role": "user", "content": "Extract invoice fields."},
    ]

    runtime.complete(messages)

    assert tokenizer.chat_messages == [
        {
            "role": "system",
            "content": "Return valid JSON only.\n\nYou extract structured fields.",
        },
        {"role": "user", "content": "Extract invoice fields."},
    ]


def test_transformers_runtime_falls_back_without_chat_template():
    runtime = TransformersRuntime(RuntimeConfig(runtime="transformers", model="demo"))
    tokenizer = PlainTokenizer()
    model = FakeModel()
    runtime._load_generator = lambda: (tokenizer, model)  # type: ignore[method-assign]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Return JSON."},
    ]

    response = runtime.complete(messages)

    assert tokenizer.rendered_prompt == render_messages_as_text(merge_system_messages(messages))
    assert model.generate_kwargs is not None
    assert "eos_token_id" not in model.generate_kwargs
    assert response.content == "plain decoded"
    assert response.raw["prompt_format"] == "text"
    assert response.raw["template_source"] is None


def test_transformers_runtime_uses_inline_template_override():
    runtime = TransformersRuntime(
        RuntimeConfig(
            runtime="transformers",
            model="demo",
            prompt_format="chat_template",
            chat_template="{{ custom_template }}",
        )
    )
    tokenizer = ChatTemplateTokenizer()
    model = FakeModel()
    runtime._load_generator = lambda: (tokenizer, model)  # type: ignore[method-assign]

    response = runtime.complete([{"role": "user", "content": "Return JSON."}])

    assert tokenizer.template_override == "{{ custom_template }}"
    assert response.raw["prompt_format"] == "chat_template"
    assert response.raw["template_source"] == "inline"


def test_schema_prompt_uses_json_template_not_raw_json_schema():
    prompt = schema_prompt(InvoiceOutput)

    assert '"vendor": "..."' in prompt
    assert '"line_items": [' in prompt
    assert '"description": "..."' in prompt
    assert '"quantity": 0' in prompt
    assert '"title": "InvoiceOutput"' not in prompt
    assert '"$defs"' not in prompt

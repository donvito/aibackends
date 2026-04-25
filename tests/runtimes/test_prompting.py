from __future__ import annotations

import pytest

from aibackends.core.exceptions import ConfigurationError
from aibackends.core.prompting import PromptRenderer
from aibackends.core.types import RuntimeConfig


class TemplateAwareTokenizer:
    def __init__(self) -> None:
        self.chat_template = "{{ tokenizer_template }}"
        self.messages: list[dict[str, str]] | None = None
        self.chat_template_override: str | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        chat_template: str | None = None,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        self.messages = messages
        self.chat_template_override = chat_template
        return "RENDERED PROMPT"


class PlainTokenizer:
    chat_template = None


def test_prompt_renderer_loads_template_override_from_path(tmp_path):
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text("{{ custom_template }}")
    renderer = PromptRenderer(
        RuntimeConfig(
            runtime="transformers",
            model="demo",
            prompt_format="chat_template",
            chat_template_path=str(template_path),
        )
    )
    tokenizer = TemplateAwareTokenizer()

    result = renderer.render([{"role": "user", "content": "Hello"}], tokenizer=tokenizer)

    assert result.prompt == "RENDERED PROMPT"
    assert result.format_used == "chat_template"
    assert result.template_source == str(template_path)
    assert tokenizer.chat_template_override == "{{ custom_template }}"


def test_prompt_renderer_falls_back_to_text_when_auto_has_no_template():
    renderer = PromptRenderer(RuntimeConfig(runtime="transformers", model="demo"))

    result = renderer.render(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        tokenizer=PlainTokenizer(),
    )

    assert result.format_used == "text"
    assert result.prompt == "SYSTEM: You are helpful.\n\nUSER: Hello"


def test_prompt_renderer_requires_template_in_chat_template_mode():
    renderer = PromptRenderer(
        RuntimeConfig(runtime="transformers", model="demo", prompt_format="chat_template")
    )

    with pytest.raises(ConfigurationError, match="No chat template is available"):
        renderer.render([{"role": "user", "content": "Hello"}], tokenizer=PlainTokenizer())

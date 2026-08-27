from __future__ import annotations

from aibackends.core.model_manager import ModelLocation
from aibackends.core.runtimes.llamacpp import LlamaCppRuntime, sanitize_chat_template
from aibackends.core.types import RuntimeConfig

LFM_STYLE_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message.role == 'assistant' %}"
    "{% generation %}{{ message.content }}{% endgeneration %}"
    "{% endif %}"
    "{% endfor %}"
)


def test_sanitize_chat_template_strips_generation_tags() -> None:
    cleaned = sanitize_chat_template(LFM_STYLE_TEMPLATE)

    assert "generation" not in cleaned
    assert "{{ message.content }}" in cleaned


def test_sanitize_chat_template_handles_whitespace_control_variants() -> None:
    template = "a{%- generation -%}b{%+ endgeneration %}c{% generation%}d{%endgeneration %}e"

    assert sanitize_chat_template(template) == "abcde"


def test_sanitize_chat_template_leaves_plain_templates_untouched() -> None:
    template = "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    assert sanitize_chat_template(template) == template


def test_build_client_kwargs_honours_chat_format_override_for_non_gemma() -> None:
    runtime = LlamaCppRuntime(
        RuntimeConfig(
            runtime="llamacpp",
            model="lfm2.5-2.6b",
            extra_options={"chat_format": "chatml"},
        )
    )

    options = runtime._build_client_kwargs(
        ModelLocation(source="LiquidAI/LFM2.5-2.6B-GGUF", local_path="/tmp/model.gguf")
    )

    assert options["chat_format"] == "chatml"


def test_build_client_kwargs_has_no_chat_format_by_default() -> None:
    runtime = LlamaCppRuntime(RuntimeConfig(runtime="llamacpp", model="lfm2.5-2.6b"))

    options = runtime._build_client_kwargs(
        ModelLocation(source="LiquidAI/LFM2.5-2.6B-GGUF", local_path="/tmp/model.gguf")
    )

    assert "chat_format" not in options

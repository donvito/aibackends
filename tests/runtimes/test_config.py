from __future__ import annotations

import pytest

from aibackends import configure
from aibackends.core.config import get_runtime, resolve_runtime_config
from aibackends.core.types import RuntimeConfig


def test_resolve_runtime_config_merges_overrides():
    configure(runtime="stub", model="default-model", extra_options={"temperature_band": "low"})
    resolved = resolve_runtime_config(
        {"model": "override-model", "extra_options": {"batch_size": 2}}
    )
    assert resolved.runtime == "stub"
    assert resolved.model == "override-model"
    assert resolved.extra_options == {"temperature_band": "low", "batch_size": 2}


def test_get_runtime_returns_registered_runtime():
    client = get_runtime()
    assert client.__class__.__name__ == "StubRuntime"
    assert client.model_name == "stub-model"


def test_runtime_config_supports_prompt_format_and_template_path(tmp_path):
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text("{{ custom_template }}")

    config = RuntimeConfig(
        runtime="transformers",
        model="demo",
        prompt_format="chat_template",
        chat_template_path=str(template_path),
    )

    assert config.prompt_format == "chat_template"
    assert config.chat_template_path == str(template_path)


def test_runtime_config_rejects_multiple_template_sources():
    with pytest.raises(ValueError, match="Only one of `chat_template` or `chat_template_path`"):
        RuntimeConfig(
            runtime="transformers",
            model="demo",
            chat_template="{{ inline }}",
            chat_template_path="~/template.jinja",
        )

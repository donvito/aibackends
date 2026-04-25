from __future__ import annotations

import pytest

from aibackends import configure, load_config
from aibackends.core.config import get_runtime, resolve_runtime_config
from aibackends.core.registry import ModelRef
from aibackends.core.types import RuntimeConfig
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP, TRANSFORMERS, get_runtime_spec


def test_resolve_runtime_config_merges_overrides():
    configure(
        runtime=get_runtime_spec("stub"),
        model=ModelRef(name="default-model"),
        extra_options={"temperature_band": "low"},
    )
    resolved = resolve_runtime_config(
        {"model": "override-model", "extra_options": {"batch_size": 2}}
    )
    assert resolved.runtime == "stub"
    assert resolved.model == "override-model"
    assert resolved.extra_options == {"temperature_band": "low", "batch_size": 2}


def test_load_config_accepts_string_runtime_and_model_values(tmp_path):
    config_path = tmp_path / "aibackends.yml"
    config_path.write_text("runtime: stub\nmodel: stub-model\n")

    loaded = load_config(config_path)

    assert loaded.runtime == "stub"
    assert loaded.model == "stub-model"


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


def test_runtime_config_accepts_typed_runtime_and_model_refs():
    config = RuntimeConfig(
        runtime=TRANSFORMERS,  # type: ignore[arg-type]
        model=GEMMA4_E2B,  # type: ignore[arg-type]
    )

    assert config.runtime == "transformers"
    assert config.model == "gemma4-e2b"


def test_resolve_runtime_config_accepts_typed_runtime_and_model_refs():
    resolved = resolve_runtime_config({"runtime": LLAMACPP, "model": GEMMA4_E2B})

    assert resolved.runtime == "llamacpp"
    assert resolved.model == "gemma4-e2b"


def test_configure_rejects_string_runtime_and_model_refs():
    with pytest.raises(TypeError, match="runtime must be a RuntimeSpec"):
        configure(runtime="stub")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="model must be a ModelRef"):
        configure(model="stub-model")  # type: ignore[arg-type]


def test_runtime_config_rejects_multiple_template_sources():
    with pytest.raises(ValueError, match="Only one of `chat_template` or `chat_template_path`"):
        RuntimeConfig(
            runtime="transformers",
            model="demo",
            chat_template="{{ inline }}",
            chat_template_path="~/template.jinja",
        )

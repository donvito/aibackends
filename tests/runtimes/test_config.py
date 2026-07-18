from __future__ import annotations

import pytest

from aibackends import configure, load_config, reset_config
from aibackends.core.config import (
    clear_runtime_cache,
    get_runtime,
    preload,
    resolve_runtime_config,
)
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


def test_get_runtime_reuses_cached_instance():
    first = get_runtime()
    second = get_runtime()

    assert first is second


def test_get_runtime_builds_new_instance_for_different_model():
    first = get_runtime()
    second = get_runtime({"model": "other-model"})

    assert first is not second


def test_get_runtime_opt_out_returns_fresh_instances():
    cached = get_runtime()
    first = get_runtime({"reuse_runtime": False})
    second = get_runtime({"reuse_runtime": False})

    assert first is not second
    assert first is not cached
    # Opting out must not overwrite the cached instance either.
    assert get_runtime() is cached


def test_get_runtime_cache_hit_refreshes_per_call_params():
    first = get_runtime({"temperature": 0.2})
    second = get_runtime({"temperature": 0.9, "max_tokens": 42})

    assert first is second
    assert second.config.temperature == 0.9
    assert second.config.max_tokens == 42


def test_clear_runtime_cache_forces_rebuild():
    first = get_runtime()
    clear_runtime_cache()

    assert get_runtime() is not first


def test_reset_config_clears_runtime_cache():
    first = get_runtime()
    reset_config()
    configure(runtime=get_runtime_spec("stub"), model=ModelRef(name="stub-model"))

    assert get_runtime() is not first


def test_preload_returns_cached_runtime():
    runtime = preload()

    assert runtime is get_runtime()


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

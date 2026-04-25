from __future__ import annotations

from aibackends.core.config import get_runtime
from aibackends.core.model_registry import register_model_profile
from aibackends.core.registry import TransformerModelProfile
from aibackends.core.runtimes.transformers import TransformersRuntime
from aibackends.core.types import RuntimeConfig
from aibackends.backends.pii import get_pii_backend
from aibackends.tasks import BaseTask, create_task
from aibackends.tasks.registry import get_task
from aibackends.workflows import Pipeline, create_workflow
from aibackends.workflows.registry import get_workflow


def test_builtin_runtime_is_discovered_from_runtime_spec():
    client = get_runtime(RuntimeConfig(runtime="ollama", model="demo"))

    assert client.__class__.__name__ == "OllamaRuntime"


def test_transformer_model_profile_can_supply_chat_template():
    register_model_profile(
        TransformerModelProfile(
            name="template-demo",
            model_id="hf/template-demo",
            chat_template="{{ messages }}",
            runtime="transformers",
        )
    )

    runtime = TransformersRuntime(RuntimeConfig(runtime="transformers", model="template-demo"))

    assert runtime.config.model == "hf/template-demo"
    assert runtime.config.chat_template == "{{ messages }}"


def test_transformer_model_profile_does_not_override_explicit_template():
    register_model_profile(
        TransformerModelProfile(
            name="explicit-template-demo",
            model_id="hf/explicit-template-demo",
            chat_template="{{ profile_template }}",
            runtime="transformers",
        )
    )

    runtime = TransformersRuntime(
        RuntimeConfig(
            runtime="transformers",
            model="explicit-template-demo",
            chat_template="{{ user_template }}",
        )
    )

    assert runtime.config.chat_template == "{{ user_template }}"


def test_pii_backend_is_discovered_from_backend_spec():
    backend = get_pii_backend("openai_privacy")

    assert backend.name == "openai-privacy"
    assert backend.model_id == "openai/privacy-filter"


def test_task_and_workflow_specs_are_discovered():
    task = get_task("extract_invoice")
    workflow = get_workflow("sales-call-analyser")

    assert task.name == "extract-invoice"
    assert workflow.name == "sales-call"
    assert workflow.pipeline_cls.__name__ == "SalesCallAnalyser"


def test_task_spec_exposes_base_task_interface():
    spec = get_task("summarize")
    task = spec.create()

    assert isinstance(task, BaseTask)
    assert callable(task.run)
    assert spec.run.__self__ is not task


def test_create_task_returns_configured_task_instance():
    task = create_task(
        "classify",
        labels=["invoice", "contract", "receipt"],
        runtime="stub",
        model="configured-model",
    )

    result = task.run("invoice for April services")

    assert result.label == "invoice"
    assert task.defaults["model"] == "configured-model"


def test_create_task_returns_fresh_instances():
    first = create_task("summarize", model="first-model")
    second = create_task("summarize", model="second-model")

    assert first is not second
    assert first.defaults["model"] == "first-model"
    assert second.defaults["model"] == "second-model"
    assert create_task("summarize").defaults == {}


def test_create_workflow_returns_configured_pipeline_instance():
    workflow = create_workflow("invoice", runtime="stub", model="configured-model")

    assert isinstance(workflow, Pipeline)
    assert workflow.config.runtime == "stub"
    assert workflow.config.model == "configured-model"


def test_create_workflow_returns_fresh_instances():
    first = create_workflow("invoice", model="first-model")
    second = create_workflow("invoice", model="second-model")

    assert first is not second
    assert first.config.model == "first-model"
    assert second.config.model == "second-model"

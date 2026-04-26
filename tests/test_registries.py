from __future__ import annotations

from types import MethodType

import pytest

from aibackends.backends.pii import get_pii_backend
from aibackends.core.config import get_runtime
from aibackends.core.model_registry import register_model_profile
from aibackends.core.registry import ModelRef, TransformerModelProfile
from aibackends.core.runtimes.transformers import TransformersRuntime
from aibackends.core.types import RuntimeConfig
from aibackends.models import GEMMA4_E2B, available_models, get_model_ref
from aibackends.runtimes import LLAMACPP, TRANSFORMERS, available_runtimes, get_runtime_spec
from aibackends.tasks import (
    BaseTask,
    ClassifyTask,
    ExtractInvoiceTask,
    SummarizeTask,
    available_tasks,
    create_task,
)
from aibackends.tasks.registry import get_task
from aibackends.workflows import (
    InvoiceProcessor,
    Pipeline,
    SalesCallAnalyser,
    available_workflows,
    create_workflow,
)
from aibackends.workflows.registry import get_workflow


def test_builtin_runtime_is_discovered_from_runtime_spec():
    client = get_runtime(RuntimeConfig(runtime="transformers", model="demo"))

    assert isinstance(client, TransformersRuntime)


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


def test_runtime_and_model_catalogs_are_discoverable():
    runtimes = available_runtimes()
    models = available_models()

    assert set(runtimes) <= {"llamacpp", "transformers", "stub"}
    assert {"llamacpp", "transformers"} <= set(runtimes)
    assert runtimes["llamacpp"] is LLAMACPP
    assert runtimes["transformers"] is TRANSFORMERS
    assert get_runtime_spec("llamacpp") is LLAMACPP
    assert get_runtime_spec("transformers") is TRANSFORMERS
    assert models["gemma4-e2b"] == GEMMA4_E2B
    assert get_model_ref("gemma4-e2b") == GEMMA4_E2B


def test_task_and_workflow_specs_are_discovered():
    task = get_task("extract_invoice")
    workflow = get_workflow("sales-call-analyser")

    assert task.name == "extract-invoice"
    assert workflow.name == "sales-call"
    assert workflow.pipeline_cls.__name__ == "SalesCallAnalyser"


def test_available_tasks_returns_canonical_names_mapped_to_task_classes():
    tasks = available_tasks()

    assert tasks["summarize"] is SummarizeTask
    assert tasks["extract-invoice"] is ExtractInvoiceTask
    assert "extract_invoice" not in tasks


def test_available_workflows_returns_canonical_names_mapped_to_pipeline_classes():
    workflows = available_workflows()

    assert workflows["invoice"] is InvoiceProcessor
    assert workflows["sales-call"] is SalesCallAnalyser
    assert "sales-call-analyser" not in workflows


def test_task_spec_exposes_base_task_interface():
    spec = get_task("summarize")
    task = spec.create()

    assert isinstance(task, BaseTask)
    assert callable(task.run)
    spec_run = spec.run
    assert isinstance(spec_run, MethodType)
    assert spec_run.__self__ is not task


def test_create_task_returns_configured_task_instance():
    task = create_task(
        ClassifyTask,
        labels=["invoice", "contract", "receipt"],
        runtime=get_runtime_spec("stub"),
        model=ModelRef(name="configured-model"),
    )

    result = task.run("invoice for April services")

    assert result.label == "invoice"
    assert task.defaults["model"] == ModelRef(name="configured-model")


def test_create_task_returns_fresh_instances():
    first = create_task(SummarizeTask, model=ModelRef(name="first-model"))
    second = create_task(SummarizeTask, model=ModelRef(name="second-model"))

    assert first is not second
    assert first.defaults["model"] == ModelRef(name="first-model")
    assert second.defaults["model"] == ModelRef(name="second-model")
    assert create_task(SummarizeTask).defaults == {}


def test_create_task_accepts_task_spec():
    task = create_task(get_task("summarize"), model=ModelRef(name="spec-model"))

    assert isinstance(task, BaseTask)
    assert task.defaults["model"] == ModelRef(name="spec-model")


def test_create_task_rejects_string_lookup():
    with pytest.raises(TypeError):
        create_task("summarize")  # type: ignore[arg-type]


def test_create_task_rejects_string_runtime_and_model_refs():
    with pytest.raises(TypeError, match="runtime must be a RuntimeSpec"):
        create_task(SummarizeTask, runtime="stub")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="model must be a ModelRef"):
        create_task(SummarizeTask, model="stub-model")  # type: ignore[arg-type]


def test_create_workflow_returns_configured_pipeline_instance():
    workflow = create_workflow(
        InvoiceProcessor,
        runtime=get_runtime_spec("stub"),
        model=ModelRef(name="configured-model"),
    )

    assert isinstance(workflow, Pipeline)
    assert workflow.config.runtime == "stub"
    assert workflow.config.model == "configured-model"


def test_create_workflow_returns_fresh_instances():
    first = create_workflow(InvoiceProcessor, model=ModelRef(name="first-model"))
    second = create_workflow(InvoiceProcessor, model=ModelRef(name="second-model"))

    assert first is not second
    assert first.config.model == "first-model"
    assert second.config.model == "second-model"


def test_create_workflow_accepts_workflow_spec():
    workflow = create_workflow(get_workflow("invoice"), model=ModelRef(name="spec-model"))

    assert isinstance(workflow, Pipeline)
    assert workflow.config.model == "spec-model"


def test_create_workflow_rejects_string_lookup():
    with pytest.raises(TypeError):
        create_workflow("invoice")  # type: ignore[arg-type]


def test_create_workflow_rejects_string_runtime_and_model_refs():
    with pytest.raises(TypeError, match="runtime must be a RuntimeSpec"):
        create_workflow(InvoiceProcessor, runtime="stub")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="model must be a ModelRef"):
        create_workflow(InvoiceProcessor, model="stub-model")  # type: ignore[arg-type]

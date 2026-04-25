from __future__ import annotations

from typing import Any

from aibackends.core.config import ensure_model_ref, ensure_runtime_spec
from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import (
    ModelRef,
    RuntimeSpec,
    WorkflowSpec,
    discover_specs,
    normalize_name,
    register_spec,
)
from aibackends.workflows._base import Pipeline

_WORKFLOWS: dict[str, WorkflowSpec] = {}
_BUILTINS_REGISTERED = False


def register_workflow(spec: WorkflowSpec) -> None:
    register_spec(_WORKFLOWS, spec, spec.names)


def get_workflow(name: str) -> WorkflowSpec:
    _ensure_builtin_workflows_registered()
    try:
        return _WORKFLOWS[normalize_name(name)]
    except KeyError as exc:
        raise TaskExecutionError(f"Unknown workflow: {name}") from exc


def create_workflow(
    workflow: type[Pipeline] | WorkflowSpec,
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **config: Any,
) -> Pipeline:
    if "runtime" in config:
        runtime = ensure_runtime_spec(config.pop("runtime"))
    else:
        runtime = ensure_runtime_spec(runtime)
    if "model" in config:
        model = ensure_model_ref(config.pop("model"))
    else:
        model = ensure_model_ref(model)
    resolved_config = {
        **config,
        "runtime": runtime,
        "model": model,
    }
    if isinstance(workflow, WorkflowSpec):
        return workflow.create(**resolved_config)
    if isinstance(workflow, type) and issubclass(workflow, Pipeline):
        return workflow(**resolved_config)
    raise TypeError(
        "create_workflow() expects a Pipeline subclass or WorkflowSpec. "
        "Use get_workflow(name) to resolve string workflow names."
    )


def available_workflows() -> dict[str, type[Pipeline]]:
    return {name: get_workflow(name).pipeline_cls for name in list_workflows()}


def list_workflows() -> list[str]:
    _ensure_builtin_workflows_registered()
    return sorted({spec.name for spec in _WORKFLOWS.values()})


def _ensure_builtin_workflows_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for spec in discover_specs("aibackends.workflows", "WORKFLOW_SPEC"):
        if not isinstance(spec, WorkflowSpec):
            raise TaskExecutionError(f"Invalid workflow spec: {spec!r}")
        register_workflow(spec)
    _BUILTINS_REGISTERED = True


__all__ = [
    "available_workflows",
    "create_workflow",
    "get_workflow",
    "list_workflows",
    "register_workflow",
]

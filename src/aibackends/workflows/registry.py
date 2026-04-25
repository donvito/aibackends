from __future__ import annotations

from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import WorkflowSpec, discover_specs, normalize_name, register_spec
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


def create_workflow(name: str, **config) -> Pipeline:
    return get_workflow(name).create(**config)


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
    "create_workflow",
    "get_workflow",
    "list_workflows",
    "register_workflow",
]

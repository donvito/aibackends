from __future__ import annotations

from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import TaskSpec, discover_specs, normalize_name, register_spec
from aibackends.tasks._base import BaseTask

_TASKS: dict[str, TaskSpec] = {}
_BUILTINS_REGISTERED = False


def register_task(spec: TaskSpec) -> None:
    register_spec(_TASKS, spec, spec.names)


def get_task(name: str) -> TaskSpec:
    _ensure_builtin_tasks_registered()
    try:
        return _TASKS[normalize_name(name)]
    except KeyError as exc:
        raise TaskExecutionError(f"Unknown task: {name}") from exc


def create_task(name: str, **config) -> BaseTask:
    return get_task(name).create(**config)


def list_tasks() -> list[str]:
    _ensure_builtin_tasks_registered()
    return sorted({spec.name for spec in _TASKS.values()})


def _ensure_builtin_tasks_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for spec in discover_specs("aibackends.tasks", "TASK_SPEC"):
        if not isinstance(spec, TaskSpec):
            raise TaskExecutionError(f"Invalid task spec: {spec!r}")
        register_task(spec)
    _BUILTINS_REGISTERED = True


__all__ = [
    "create_task",
    "get_task",
    "list_tasks",
    "register_task",
]

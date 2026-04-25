from __future__ import annotations

from typing import Any

from aibackends.core.config import ensure_model_ref, ensure_runtime_spec
from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import (
    ModelRef,
    RuntimeSpec,
    TaskSpec,
    discover_specs,
    normalize_name,
    register_spec,
)
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


def create_task(
    task: type[BaseTask] | TaskSpec,
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **config: Any,
) -> BaseTask:
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
    if isinstance(task, TaskSpec):
        return task.create(**resolved_config)
    if isinstance(task, type) and issubclass(task, BaseTask):
        return task(**resolved_config)
    raise TypeError(
        "create_task() expects a BaseTask subclass or TaskSpec. "
        "Use get_task(name) to resolve string task names."
    )


def available_tasks() -> dict[str, type[BaseTask]]:
    return {name: _task_class_for_spec(get_task(name)) for name in list_tasks()}


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


def _task_class_for_spec(spec: TaskSpec) -> type[BaseTask]:
    task_factory = spec.task_factory
    if isinstance(task_factory, type) and issubclass(task_factory, BaseTask):
        return task_factory
    raise TaskExecutionError(f"Task spec {spec.name!r} does not expose a BaseTask subclass.")


__all__ = [
    "available_tasks",
    "create_task",
    "get_task",
    "list_tasks",
    "register_task",
]

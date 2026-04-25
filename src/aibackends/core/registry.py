from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from aibackends.core.model_manager import ModelLocation, ModelManager
    from aibackends.core.runtimes.base import BaseRuntime
    from aibackends.core.types import RuntimeConfig
    from aibackends.schemas.pii import PIIEntity
    from aibackends.tasks._base import BaseTask
    from aibackends.workflows._base import Pipeline

RuntimeFactory = Callable[["RuntimeConfig"], "BaseRuntime"]
TaskFactory = Callable[..., "BaseTask"]
WorkflowFactory = type["Pipeline"]
PIIDetector = Callable[["PIIBackendSpec", str, list[str] | None], list["PIIEntity"]]
ModelSupportHandler = Callable[
    ["ModelManager", "RuntimeConfig", str],
    "ModelLocation",
]

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RuntimeSpec:
    name: str
    factory: RuntimeFactory
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True, slots=True)
class ModelRef:
    name: str


@dataclass(frozen=True, slots=True)
class TransformerModelProfile:
    name: str
    model_id: str
    aliases: tuple[str, ...] = ()
    runtime: str | None = "transformers"
    prompt_format: str | None = None
    chat_template: str | None = None
    chat_template_path: str | None = None
    generation_defaults: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True, slots=True)
class PIIBackendSpec:
    name: str
    detect: PIIDetector
    aliases: tuple[str, ...] = ()
    model_id: str | None = None
    default_labels: tuple[str, ...] = ()
    threshold: float | None = None
    supports_custom_labels: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass(frozen=True, slots=True)
class ModelSupportSpec:
    runtime: str
    ensure_model: ModelSupportHandler | None = None
    pull_model: ModelSupportHandler | None = None


@dataclass(frozen=True, slots=True)
class TaskSpec:
    name: str
    task_factory: TaskFactory
    aliases: tuple[str, ...] = ()
    accepts_runtime: bool = True
    accepts_model: bool = True
    accepts_backend: bool = False
    accepts_labels: bool = False
    requires_labels: bool = False
    requires_schema: bool = False

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def create(self, **config: Any) -> BaseTask:
        return self.task_factory(**config)

    @property
    def task(self) -> BaseTask:
        return self.create()

    @property
    def run(self) -> Callable[..., Any]:
        return self.create().run

    @property
    def run_async(self) -> Callable[..., Any]:
        return self.create().run_async


@dataclass(frozen=True, slots=True)
class WorkflowSpec:
    name: str
    workflow_factory: WorkflowFactory
    aliases: tuple[str, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)

    def create(self, **config: Any) -> Pipeline:
        return self.workflow_factory(**config)

    @property
    def pipeline_cls(self) -> type[Pipeline]:
        return self.workflow_factory


def normalize_name(name: str) -> str:
    return name.replace("_", "-").lower()


def discover_specs(package_name: str, attr_name: str) -> list[Any]:
    specs: list[Any] = []
    for module in iter_package_modules(package_name):
        if hasattr(module, attr_name):
            specs.extend(_flatten_specs(getattr(module, attr_name)))
    return specs


def iter_package_modules(package_name: str) -> Iterable[ModuleType]:
    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return []

    modules: list[ModuleType] = []
    prefix = f"{package.__name__}."
    for module_info in pkgutil.iter_modules(package_path, prefix):
        short_name = module_info.name.rsplit(".", 1)[-1]
        if short_name.startswith("_") or short_name == "base":
            continue
        modules.append(importlib.import_module(module_info.name))
    return modules


def register_spec(registry: dict[str, T], spec: T, names: Iterable[str]) -> None:
    for name in names:
        registry[normalize_name(name)] = spec


def _flatten_specs(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [item for item in value.values() if item is not None]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [item for item in value if item is not None]
    return [value]


RuntimeRefLike = RuntimeSpec | str
ModelRefLike = ModelRef | str

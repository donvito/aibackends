from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from typing import Any

from aibackends.core.config import ensure_model_ref, ensure_runtime_spec


class BaseTask(ABC):
    name = "task"

    def __init__(self, **defaults: Any) -> None:
        if "runtime" in defaults:
            defaults["runtime"] = ensure_runtime_spec(defaults["runtime"])
        if "model" in defaults:
            defaults["model"] = ensure_model_ref(defaults["model"])
        self.defaults = {key: value for key, value in defaults.items() if value is not None}

    def with_config(self, **defaults: Any) -> BaseTask:
        task = copy.copy(self)
        task.defaults = self._resolve_options(**defaults)
        return task

    @abstractmethod
    def run(self, input: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def run_async(self, input: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self.run, input, **kwargs)

    def _resolve_options(self, **overrides: Any) -> dict[str, Any]:
        options = dict(self.defaults)
        options.update({key: value for key, value in overrides.items() if value is not None})
        if "runtime" in options:
            options["runtime"] = ensure_runtime_spec(options["runtime"])
        if "model" in options:
            options["model"] = ensure_model_ref(options["model"])
        return options

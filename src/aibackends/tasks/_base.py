from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from typing import Any


class BaseTask(ABC):
    name = "task"

    def __init__(self, **defaults: Any) -> None:
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
        return options

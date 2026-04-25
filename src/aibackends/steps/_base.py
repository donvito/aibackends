from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from aibackends.core.types import RuntimeConfig


@dataclass(frozen=True, slots=True)
class StepContext:
    task_name: str
    runtime_config: RuntimeConfig


class BaseStep(ABC):
    name = "step"

    @abstractmethod
    def run(self, payload: Any, context: StepContext) -> Any:
        raise NotImplementedError

    async def run_async(self, payload: Any, context: StepContext) -> Any:
        return await asyncio.to_thread(self.run, payload, context)

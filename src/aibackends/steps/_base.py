from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class BaseStep(ABC):
    name = "step"

    @abstractmethod
    def run(self, payload: Any, context: dict[str, Any]) -> Any:
        raise NotImplementedError

    async def run_async(self, payload: Any, context: dict[str, Any]) -> Any:
        return await asyncio.to_thread(self.run, payload, context)

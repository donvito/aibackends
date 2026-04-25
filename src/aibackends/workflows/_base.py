from __future__ import annotations

from typing import Any

from aibackends.core.assembler import Assembler
from aibackends.core.config import resolve_runtime_config
from aibackends.core.types import BatchRunResult
from aibackends.steps._base import BaseStep


class Pipeline:
    steps: list[BaseStep] = []

    def __init__(
        self, *, runtime: str | None = None, model: str | None = None, **overrides: Any
    ) -> None:
        self.config = resolve_runtime_config({"runtime": runtime, "model": model, **overrides})
        self.steps = list(self.steps)
        self.assembler = Assembler(
            self.steps, task_name=self.__class__.__name__, config=self.config
        )

    def run(self, payload: Any) -> Any:
        return self.assembler.run(payload)

    async def run_async(self, payload: Any) -> Any:
        return await self.assembler.run_async(payload)

    def run_batch(
        self,
        *,
        inputs,
        max_concurrency: int = 4,
        on_error: str = "raise",
    ) -> BatchRunResult:
        return self.assembler.run_batch(inputs, max_concurrency=max_concurrency, on_error=on_error)

    async def run_batch_async(
        self,
        *,
        inputs,
        max_concurrency: int = 4,
        on_error: str = "raise",
    ) -> BatchRunResult:
        return await self.assembler.run_batch_async(
            inputs,
            max_concurrency=max_concurrency,
            on_error=on_error,
        )

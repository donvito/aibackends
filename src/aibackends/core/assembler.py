from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from aibackends.core.exceptions import WorkflowStepError
from aibackends.core.logging import emit_step_log
from aibackends.core.types import BatchError, BatchRunResult, RuntimeConfig, StepLog

if TYPE_CHECKING:
    from aibackends.steps._base import BaseStep, StepContext


class Assembler:
    def __init__(
        self, steps: list[BaseStep], task_name: str, config: RuntimeConfig | None = None
    ) -> None:
        self.steps = steps
        self.task_name = task_name
        self.config = config or RuntimeConfig()

    def run(self, payload: Any) -> Any:
        context = self._step_context()
        current = payload
        for step in self.steps:
            current = self._run_step(step, current, context)
        return current

    async def run_async(self, payload: Any) -> Any:
        context = self._step_context()
        current = payload
        for step in self.steps:
            current = await self._run_step_async(step, current, context)
        return current

    def run_batch(
        self,
        inputs: Iterable[Any],
        *,
        max_concurrency: int = 4,
        on_error: str = "raise",
    ) -> BatchRunResult:
        results: list[Any] = []
        errors: list[BatchError] = []
        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            futures = {executor.submit(self.run, item): item for item in inputs}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    if on_error == "collect":
                        errors.append(
                            BatchError(
                                input_path=str(item),
                                step_name=self.task_name,
                                exception=str(exc),
                            )
                        )
                    if on_error not in {"skip", "collect"}:
                        raise ValueError("on_error must be one of: raise, skip, collect") from exc
        return BatchRunResult(results=results, errors=errors)

    async def run_batch_async(
        self,
        inputs: Iterable[Any],
        *,
        max_concurrency: int = 4,
        on_error: str = "raise",
    ) -> BatchRunResult:
        semaphore = asyncio.Semaphore(max_concurrency)
        results: list[Any] = []
        errors: list[BatchError] = []

        async def runner(item: Any) -> None:
            async with semaphore:
                try:
                    results.append(await self.run_async(item))
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    if on_error == "collect":
                        errors.append(
                            BatchError(
                                input_path=str(item),
                                step_name=self.task_name,
                                exception=str(exc),
                            )
                        )
                    if on_error not in {"skip", "collect"}:
                        raise ValueError("on_error must be one of: raise, skip, collect") from exc

        await asyncio.gather(*(runner(item) for item in inputs))
        return BatchRunResult(results=results, errors=errors)

    def _step_context(self) -> StepContext:
        from aibackends.steps._base import StepContext

        return StepContext(task_name=self.task_name, runtime_config=self.config)

    def _run_step(self, step: BaseStep, payload: Any, context: StepContext) -> Any:
        started = time.perf_counter()
        emit_step_log(
            StepLog(task_name=self.task_name, step_name=step.name, status="started"), self.config
        )
        try:
            result = step.run(payload, context)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            emit_step_log(
                StepLog(
                    task_name=self.task_name,
                    step_name=step.name,
                    status="failed",
                    elapsed_ms=elapsed_ms,
                    metadata={"error": str(exc)},
                ),
                self.config,
            )
            raise WorkflowStepError(f"{self.task_name}:{step.name} failed: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        emit_step_log(
            StepLog(
                task_name=self.task_name,
                step_name=step.name,
                status="completed",
                elapsed_ms=elapsed_ms,
            ),
            self.config,
        )
        return result

    async def _run_step_async(self, step: BaseStep, payload: Any, context: StepContext) -> Any:
        started = time.perf_counter()
        emit_step_log(
            StepLog(task_name=self.task_name, step_name=step.name, status="started"), self.config
        )
        try:
            result = await step.run_async(payload, context)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            emit_step_log(
                StepLog(
                    task_name=self.task_name,
                    step_name=step.name,
                    status="failed",
                    elapsed_ms=elapsed_ms,
                    metadata={"error": str(exc)},
                ),
                self.config,
            )
            raise WorkflowStepError(f"{self.task_name}:{step.name} failed: {exc}") from exc
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        emit_step_log(
            StepLog(
                task_name=self.task_name,
                step_name=step.name,
                status="completed",
                elapsed_ms=elapsed_ms,
            ),
            self.config,
        )
        return result

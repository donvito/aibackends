from __future__ import annotations

import asyncio
from typing import Any

from aibackends.core.config import ensure_model_ref, ensure_runtime_spec
from aibackends.core.registry import ModelRef, RuntimeSpec, TaskSpec
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import run_embedding_task


def embed(
    text: str,
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **overrides: Any,
) -> list[float]:
    runtime = ensure_runtime_spec(runtime)
    model = ensure_model_ref(model)
    return run_embedding_task(
        task_name="embed",
        text=text,
        runtime=runtime,
        model=model,
        **overrides,
    )


async def embed_async(
    text: str,
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **overrides: Any,
) -> list[float]:
    return await asyncio.to_thread(embed, text, runtime=runtime, model=model, **overrides)


class EmbedTask(BaseTask):
    name = "embed"

    def run(
        self,
        input: str,
        *,
        runtime: RuntimeSpec | None = None,
        model: ModelRef | None = None,
        **overrides: Any,
    ) -> list[float]:
        options = self._resolve_options(runtime=runtime, model=model, **overrides)
        return embed(input, **options)


TASK_SPEC = TaskSpec(name=EmbedTask.name, task_factory=EmbedTask)

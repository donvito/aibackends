from __future__ import annotations

import asyncio
from typing import Any

from aibackends.core.registry import TaskSpec
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import run_embedding_task


def embed(
    text: str,
    *,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> list[float]:
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
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> list[float]:
    return await asyncio.to_thread(embed, text, runtime=runtime, model=model, **overrides)


class EmbedTask(BaseTask):
    name = "embed"

    def run(
        self,
        input: str,
        *,
        runtime: str | None = None,
        model: str | None = None,
        **overrides: Any,
    ) -> list[float]:
        options = self._resolve_options(runtime=runtime, model=model, **overrides)
        return embed(input, **options)


TASK_SPEC = TaskSpec(name=EmbedTask.name, task_factory=EmbedTask)

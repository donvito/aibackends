from __future__ import annotations

import asyncio
from typing import Any

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

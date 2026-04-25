from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from aibackends.core.registry import TaskSpec
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import build_messages, load_text_input, run_structured_task

T = TypeVar("T", bound=BaseModel)


def extract(
    text: str | Path,
    *,
    schema: type[T],
    instructions: str | None = None,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> T:
    content = load_text_input(text)
    prompt = instructions or "Extract the requested structured information from the content."
    messages = build_messages(
        "You are a high-accuracy information extraction engine.",
        f"{prompt}\n\nContent:\n{content}",
    )
    return run_structured_task(
        task_name="extract",
        schema=schema,
        messages=messages,
        runtime=runtime,
        model=model,
        **overrides,
    )


async def extract_async(
    text: str | Path,
    *,
    schema: type[T],
    instructions: str | None = None,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> T:
    def _run() -> T:
        return extract(
            text,
            schema=schema,
            instructions=instructions,
            runtime=runtime,
            model=model,
            **overrides,
        )

    return await asyncio.to_thread(_run)


class ExtractTask(BaseTask):
    name = "extract"

    def run(
        self,
        input: str | Path,
        *,
        schema: type[T] | None = None,
        instructions: str | None = None,
        runtime: str | None = None,
        model: str | None = None,
        **overrides: Any,
    ) -> T:
        options = self._resolve_options(
            schema=schema,
            instructions=instructions,
            runtime=runtime,
            model=model,
            **overrides,
        )
        return extract(input, **options)


TASK_SPEC = TaskSpec(
    name=ExtractTask.name,
    task_factory=ExtractTask,
    requires_schema=True,
)

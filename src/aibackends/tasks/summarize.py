from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.core.config import ensure_model_ref, ensure_runtime_spec
from aibackends.core.registry import ModelRef, RuntimeSpec, TaskSpec
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import build_messages, load_text_input, run_text_task


def summarize(
    text: str | Path,
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **overrides: Any,
) -> str:
    runtime = ensure_runtime_spec(runtime)
    model = ensure_model_ref(model)
    content = load_text_input(text)
    messages = build_messages(
        "You are an expert summarization engine.",
        f"Summarize the following content in clear, concise prose.\n\n{content}",
    )
    return run_text_task(
        task_name="summarize",
        messages=messages,
        runtime=runtime,
        model=model,
        **overrides,
    )


async def summarize_async(
    text: str | Path,
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **overrides: Any,
) -> str:
    return await asyncio.to_thread(summarize, text, runtime=runtime, model=model, **overrides)


class SummarizeTask(BaseTask):
    name = "summarize"

    def run(
        self,
        input: str | Path,
        *,
        runtime: RuntimeSpec | None = None,
        model: ModelRef | None = None,
        **overrides: Any,
    ) -> str:
        options = self._resolve_options(runtime=runtime, model=model, **overrides)
        return summarize(input, **options)


TASK_SPEC = TaskSpec(name=SummarizeTask.name, task_factory=SummarizeTask)

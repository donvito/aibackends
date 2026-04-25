from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.tasks._utils import build_messages, load_text_input, run_text_task


def summarize(
    text: str | Path,
    *,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> str:
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
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> str:
    return await asyncio.to_thread(summarize, text, runtime=runtime, model=model, **overrides)

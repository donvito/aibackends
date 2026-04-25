from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.schemas.pii import Classification
from aibackends.tasks._utils import build_messages, load_text_input, run_structured_task


def classify(
    text: str | Path,
    *,
    labels: list[str],
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> Classification:
    content = load_text_input(text)
    label_list = ", ".join(labels)
    messages = build_messages(
        "You are a precise text classification engine.",
        (
            "Classify the text into exactly one of these labels: "
            f"{label_list}. Return the winning label, a confidence score between 0 and 1, "
            "and scores for every provided label.\n\n"
            f"Text:\n{content}"
        ),
    )
    return run_structured_task(
        task_name="classify",
        schema=Classification,
        messages=messages,
        runtime=runtime,
        model=model,
        **overrides,
    )


async def classify_async(
    text: str | Path,
    *,
    labels: list[str],
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> Classification:
    return await asyncio.to_thread(
        classify,
        text,
        labels=labels,
        runtime=runtime,
        model=model,
        **overrides,
    )

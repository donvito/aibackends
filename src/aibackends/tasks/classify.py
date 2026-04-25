from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.core.registry import TaskSpec
from aibackends.schemas.pii import Classification
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import build_messages, load_text_input, run_structured_task

DEFAULT_SYSTEM_PROMPT = "You are a precise text classification engine."


def _build_classification_prompt(
    text: str,
    *,
    labels: list[str],
    label_descriptions: dict[str, str] | None = None,
    prompt: str | None = None,
) -> str:
    label_list = ", ".join(labels)
    parts: list[str] = []
    if prompt:
        parts.append(prompt.strip())
    parts.append(
        "Classify the text into exactly one of these labels: "
        f"{label_list}. Return the winning label, a confidence score between 0 and 1, "
        "and scores for every provided label."
    )
    if label_descriptions:
        unknown = sorted(set(label_descriptions) - set(labels))
        if unknown:
            unknown_list = ", ".join(unknown)
            raise ValueError(f"label_descriptions contains unknown labels: {unknown_list}")
        described_labels = "\n".join(
            f"- {label}: {label_descriptions[label]}"
            for label in labels
            if label in label_descriptions
        )
        if described_labels:
            parts.append(f"Label descriptions:\n{described_labels}")
    parts.append(f"Text:\n{text}")
    return "\n\n".join(parts)


def classify(
    text: str | Path,
    *,
    labels: list[str],
    label_descriptions: dict[str, str] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt: str | None = None,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> Classification:
    content = load_text_input(text)
    messages = build_messages(
        system_prompt,
        _build_classification_prompt(
            content,
            labels=labels,
            label_descriptions=label_descriptions,
            prompt=prompt,
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
    label_descriptions: dict[str, str] | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt: str | None = None,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> Classification:
    return await asyncio.to_thread(
        classify,
        text,
        labels=labels,
        label_descriptions=label_descriptions,
        system_prompt=system_prompt,
        prompt=prompt,
        runtime=runtime,
        model=model,
        **overrides,
    )


class ClassifyTask(BaseTask):
    name = "classify"

    def run(
        self,
        input: str | Path,
        *,
        labels: list[str] | None = None,
        label_descriptions: dict[str, str] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        prompt: str | None = None,
        runtime: str | None = None,
        model: str | None = None,
        **overrides: Any,
    ) -> Classification:
        options = self._resolve_options(
            labels=labels,
            label_descriptions=label_descriptions,
            system_prompt=system_prompt,
            prompt=prompt,
            runtime=runtime,
            model=model,
            **overrides,
        )
        return classify(input, **options)


TASK_SPEC = TaskSpec(
    name=ClassifyTask.name,
    task_factory=ClassifyTask,
    accepts_labels=True,
    requires_labels=True,
)

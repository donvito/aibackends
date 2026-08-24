from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aibackends.backends.moderation import get_moderation_backend
from aibackends.core.registry import TaskSpec
from aibackends.schemas.moderation import PromptModeration, ResponseModeration
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import load_text_input


def moderate_prompt(
    prompt: str | Path,
    *,
    backend: str = "gliguard",
    device: str = "cpu",
    threshold: float = 0.5,
    category_threshold: float = 0.4,
) -> PromptModeration:
    content = load_text_input(prompt)
    return get_moderation_backend(backend).moderate_prompt(
        content,
        device=device,
        threshold=threshold,
        category_threshold=category_threshold,
    )


async def moderate_prompt_async(
    prompt: str | Path,
    *,
    backend: str = "gliguard",
    device: str = "cpu",
    threshold: float = 0.5,
    category_threshold: float = 0.4,
) -> PromptModeration:
    return await asyncio.to_thread(
        moderate_prompt,
        prompt,
        backend=backend,
        device=device,
        threshold=threshold,
        category_threshold=category_threshold,
    )


def moderate_prompts(
    prompts: Sequence[str | Path],
    *,
    backend: str = "gliguard",
    device: str = "cpu",
    threshold: float = 0.5,
    category_threshold: float = 0.4,
    batch_size: int = 8,
) -> list[PromptModeration]:
    contents = [load_text_input(prompt) for prompt in prompts]
    return get_moderation_backend(backend).moderate_prompts(
        contents,
        device=device,
        threshold=threshold,
        category_threshold=category_threshold,
        batch_size=batch_size,
    )


async def moderate_prompts_async(
    prompts: Sequence[str | Path],
    *,
    backend: str = "gliguard",
    device: str = "cpu",
    threshold: float = 0.5,
    category_threshold: float = 0.4,
    batch_size: int = 8,
) -> list[PromptModeration]:
    return await asyncio.to_thread(
        moderate_prompts,
        prompts,
        backend=backend,
        device=device,
        threshold=threshold,
        category_threshold=category_threshold,
        batch_size=batch_size,
    )


def moderate_response(
    response: str | Path,
    *,
    prompt: str | Path | None = None,
    backend: str = "gliguard",
    device: str = "cpu",
    threshold: float = 0.5,
    category_threshold: float = 0.4,
) -> ResponseModeration:
    response_content = load_text_input(response)
    prompt_content = load_text_input(prompt) if prompt is not None else None
    return get_moderation_backend(backend).moderate_response(
        response_content,
        prompt=prompt_content,
        device=device,
        threshold=threshold,
        category_threshold=category_threshold,
    )


async def moderate_response_async(
    response: str | Path,
    *,
    prompt: str | Path | None = None,
    backend: str = "gliguard",
    device: str = "cpu",
    threshold: float = 0.5,
    category_threshold: float = 0.4,
) -> ResponseModeration:
    return await asyncio.to_thread(
        moderate_response,
        response,
        prompt=prompt,
        backend=backend,
        device=device,
        threshold=threshold,
        category_threshold=category_threshold,
    )


def moderate_responses(
    responses: Sequence[str | Path],
    *,
    prompts: Sequence[str | Path | None] | None = None,
    backend: str = "gliguard",
    device: str = "cpu",
    threshold: float = 0.5,
    category_threshold: float = 0.4,
    batch_size: int = 8,
) -> list[ResponseModeration]:
    response_contents = [load_text_input(response) for response in responses]
    prompt_contents = (
        [load_text_input(prompt) if prompt is not None else None for prompt in prompts]
        if prompts is not None
        else None
    )
    return get_moderation_backend(backend).moderate_responses(
        response_contents,
        prompts=prompt_contents,
        device=device,
        threshold=threshold,
        category_threshold=category_threshold,
        batch_size=batch_size,
    )


async def moderate_responses_async(
    responses: Sequence[str | Path],
    *,
    prompts: Sequence[str | Path | None] | None = None,
    backend: str = "gliguard",
    device: str = "cpu",
    threshold: float = 0.5,
    category_threshold: float = 0.4,
    batch_size: int = 8,
) -> list[ResponseModeration]:
    return await asyncio.to_thread(
        moderate_responses,
        responses,
        prompts=prompts,
        backend=backend,
        device=device,
        threshold=threshold,
        category_threshold=category_threshold,
        batch_size=batch_size,
    )


class ModeratePromptTask(BaseTask):
    name = "moderate-prompt"

    def run(
        self,
        input: str | Path,
        *,
        backend: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        category_threshold: float | None = None,
        **overrides: Any,
    ) -> PromptModeration:
        options = self._resolve_options(
            backend=backend,
            device=device,
            threshold=threshold,
            category_threshold=category_threshold,
            **overrides,
        )
        options.setdefault("backend", "gliguard")
        return moderate_prompt(input, **options)

    def run_batch(
        self,
        inputs: Sequence[str | Path],
        *,
        backend: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        category_threshold: float | None = None,
        batch_size: int | None = None,
        **overrides: Any,
    ) -> list[PromptModeration]:
        options = self._resolve_options(
            backend=backend,
            device=device,
            threshold=threshold,
            category_threshold=category_threshold,
            batch_size=batch_size,
            **overrides,
        )
        options.setdefault("backend", "gliguard")
        return moderate_prompts(inputs, **options)


class ModerateResponseTask(BaseTask):
    name = "moderate-response"

    def run(
        self,
        input: str | Path,
        *,
        prompt: str | Path | None = None,
        backend: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        category_threshold: float | None = None,
        **overrides: Any,
    ) -> ResponseModeration:
        options = self._resolve_options(
            prompt=prompt,
            backend=backend,
            device=device,
            threshold=threshold,
            category_threshold=category_threshold,
            **overrides,
        )
        options.setdefault("backend", "gliguard")
        return moderate_response(input, **options)

    def run_batch(
        self,
        inputs: Sequence[str | Path],
        *,
        prompts: Sequence[str | Path | None] | None = None,
        backend: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        category_threshold: float | None = None,
        batch_size: int | None = None,
        **overrides: Any,
    ) -> list[ResponseModeration]:
        options = self._resolve_options(
            prompts=prompts,
            backend=backend,
            device=device,
            threshold=threshold,
            category_threshold=category_threshold,
            batch_size=batch_size,
            **overrides,
        )
        options.setdefault("backend", "gliguard")
        return moderate_responses(inputs, **options)


TASK_SPEC = (
    TaskSpec(
        name=ModeratePromptTask.name,
        task_factory=ModeratePromptTask,
        aliases=("moderate_prompt",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_backend=True,
        accepts_device=True,
        accepts_threshold=True,
        accepts_category_threshold=True,
    ),
    TaskSpec(
        name=ModerateResponseTask.name,
        task_factory=ModerateResponseTask,
        aliases=("moderate_response",),
        accepts_runtime=False,
        accepts_model=False,
        accepts_backend=True,
        accepts_prompt=True,
        accepts_device=True,
        accepts_threshold=True,
        accepts_category_threshold=True,
    ),
)

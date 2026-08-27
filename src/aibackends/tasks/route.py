from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aibackends.backends.routing import get_routing_backend
from aibackends.core.registry import TaskSpec
from aibackends.schemas.routing import RoutingResult
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import load_text_input

DEFAULT_ROUTING_BACKEND = "lfm2-prompt-router"


def route_prompt(
    prompt: str | Path,
    routes: Sequence[str],
    *,
    backend: str = DEFAULT_ROUTING_BACKEND,
    device: str = "cpu",
    threshold: float | None = None,
) -> RoutingResult:
    content = load_text_input(prompt)
    return get_routing_backend(backend).route(
        content,
        routes,
        device=device,
        threshold=threshold,
    )


async def route_prompt_async(
    prompt: str | Path,
    routes: Sequence[str],
    *,
    backend: str = DEFAULT_ROUTING_BACKEND,
    device: str = "cpu",
    threshold: float | None = None,
) -> RoutingResult:
    return await asyncio.to_thread(
        route_prompt,
        prompt,
        routes,
        backend=backend,
        device=device,
        threshold=threshold,
    )


def route_prompts(
    prompts: Sequence[str | Path],
    routes: Sequence[str],
    *,
    backend: str = DEFAULT_ROUTING_BACKEND,
    device: str = "cpu",
    threshold: float | None = None,
) -> list[RoutingResult]:
    contents = [load_text_input(prompt) for prompt in prompts]
    return get_routing_backend(backend).route_batch(
        contents,
        routes,
        device=device,
        threshold=threshold,
    )


async def route_prompts_async(
    prompts: Sequence[str | Path],
    routes: Sequence[str],
    *,
    backend: str = DEFAULT_ROUTING_BACKEND,
    device: str = "cpu",
    threshold: float | None = None,
) -> list[RoutingResult]:
    return await asyncio.to_thread(
        route_prompts,
        prompts,
        routes,
        backend=backend,
        device=device,
        threshold=threshold,
    )


class RoutePromptTask(BaseTask):
    name = "route-prompt"

    def run(
        self,
        input: str | Path,
        *,
        routes: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        backend: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        **overrides: Any,
    ) -> RoutingResult:
        options = self._resolve_options(
            # `labels` is the CLI spelling of `routes`; either works.
            routes=routes if routes is not None else labels,
            backend=backend,
            device=device,
            threshold=threshold,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_ROUTING_BACKEND)
        resolved_routes = options.pop("routes", None)
        if not resolved_routes:
            raise ValueError("route-prompt requires at least one route via routes/labels.")
        return route_prompt(input, resolved_routes, **options)

    def run_batch(
        self,
        inputs: Sequence[str | Path],
        *,
        routes: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        backend: str | None = None,
        device: str | None = None,
        threshold: float | None = None,
        **overrides: Any,
    ) -> list[RoutingResult]:
        options = self._resolve_options(
            routes=routes if routes is not None else labels,
            backend=backend,
            device=device,
            threshold=threshold,
            **overrides,
        )
        options.setdefault("backend", DEFAULT_ROUTING_BACKEND)
        resolved_routes = options.pop("routes", None)
        if not resolved_routes:
            raise ValueError("route-prompt requires at least one route via routes/labels.")
        return route_prompts(inputs, resolved_routes, **options)


TASK_SPEC = TaskSpec(
    name=RoutePromptTask.name,
    task_factory=RoutePromptTask,
    aliases=("route_prompt",),
    accepts_runtime=False,
    accepts_model=False,
    accepts_backend=True,
    accepts_labels=True,
    requires_labels=True,
    accepts_device=True,
    accepts_threshold=True,
)

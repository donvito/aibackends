from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.backends.pii import get_pii_backend
from aibackends.core.registry import TaskSpec
from aibackends.schemas.pii import RedactedText
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import load_text_input


def redact_pii(
    text: str | Path,
    *,
    backend: str = "gliner",
    labels: list[str] | None = None,
    **overrides: Any,
) -> RedactedText:
    del overrides
    content = load_text_input(text)
    backend_spec = get_pii_backend(backend)
    return backend_spec.redact(content, labels=labels)


async def redact_pii_async(
    text: str | Path,
    *,
    backend: str = "gliner",
    labels: list[str] | None = None,
    **overrides: Any,
) -> RedactedText:
    return await asyncio.to_thread(redact_pii, text, backend=backend, labels=labels, **overrides)


class RedactPIITask(BaseTask):
    name = "redact-pii"

    def run(
        self,
        input: str | Path,
        *,
        backend: str | None = None,
        labels: list[str] | None = None,
        **overrides: Any,
    ) -> RedactedText:
        options = self._resolve_options(backend=backend, labels=labels, **overrides)
        if "backend" not in options:
            options["backend"] = "gliner"
        return redact_pii(input, **options)


TASK_SPEC = TaskSpec(
    name=RedactPIITask.name,
    task_factory=RedactPIITask,
    aliases=("redact_pii",),
    accepts_runtime=False,
    accepts_model=False,
    accepts_backend=True,
    accepts_labels=True,
)

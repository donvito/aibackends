from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.backends.pii import get_pii_backend
from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import TaskSpec
from aibackends.schemas.pii import PIIEntity, RedactedText
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
    if labels is not None and not backend_spec.supports_custom_labels:
        raise TaskExecutionError(
            f"Custom labels are not supported for the {backend_spec.name} PII backend."
        )
    entities = backend_spec.detect(backend_spec, content, labels)

    redacted_text, redaction_map, normalized_entities = _apply_redactions(content, entities)
    return RedactedText(
        original_text=content,
        redacted_text=redacted_text,
        entities_found=normalized_entities,
        redaction_map=redaction_map,
        backend_used=backend_spec.name,
    )


async def redact_pii_async(
    text: str | Path,
    *,
    backend: str = "gliner",
    labels: list[str] | None = None,
    **overrides: Any,
) -> RedactedText:
    return await asyncio.to_thread(redact_pii, text, backend=backend, labels=labels, **overrides)


def _apply_redactions(
    text: str, entities: list[PIIEntity]
) -> tuple[str, dict[str, str], list[PIIEntity]]:
    parts: list[str] = []
    cursor = 0
    redaction_map: dict[str, str] = {}
    normalized: list[PIIEntity] = []

    for index, entity in enumerate(sorted(entities, key=lambda item: item.start), start=1):
        if entity.start < cursor:
            continue
        replacement = f"[{entity.entity_type}_{index}]"
        parts.append(text[cursor : entity.start])
        parts.append(replacement)
        cursor = entity.end
        redaction_map[entity.text] = replacement
        normalized.append(entity.model_copy(update={"replacement": replacement}))

    parts.append(text[cursor:])
    return "".join(parts), redaction_map, normalized


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

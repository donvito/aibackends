from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
import sys
from typing import Any

from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.schemas.pii import PIIEntity, RedactedText
from aibackends.tasks._utils import load_text_input

GLINER_MODEL_ID = "nvidia/gliner-pii"
GLINER_LABELS = ["email", "phone_number", "user_name"]
GLINER_THRESHOLD = 0.5
# Run GLiNER in a child process so native OpenMP/torch crashes
# don't abort the parent Python process.
GLINER_WORKER_PATH = Path(__file__).with_name("_gliner_worker.py")


def redact_pii(
    text: str | Path,
    *,
    backend: str = "gliner",
    labels: list[str] | None = None,
    **overrides: Any,
) -> RedactedText:
    del overrides
    content = load_text_input(text)
    entities = _entities_for_backend(content, backend, labels=labels)

    redacted_text, redaction_map, normalized_entities = _apply_redactions(content, entities)
    return RedactedText(
        original_text=content,
        redacted_text=redacted_text,
        entities_found=normalized_entities,
        redaction_map=redaction_map,
        backend_used=backend,
    )


async def redact_pii_async(
    text: str | Path,
    *,
    backend: str = "gliner",
    labels: list[str] | None = None,
    **overrides: Any,
) -> RedactedText:
    return await asyncio.to_thread(redact_pii, text, backend=backend, labels=labels, **overrides)


def _entities_for_backend(
    text: str, backend: str, *, labels: list[str] | None = None
) -> list[PIIEntity]:
    if backend == "gliner":
        return _gliner_entities(text, labels=labels)
    if backend == "openai-privacy":
        if labels is not None:
            raise TaskExecutionError(
                "Custom labels are only supported for the gliner PII backend."
            )
        return _privacy_filter_entities(text)
    raise TaskExecutionError(f"Unsupported PII backend: {backend}")


def _predict_gliner_raw_entities(
    text: str, *, labels: list[str] | None = None
) -> list[dict[str, Any]]:
    payload = json.dumps(
        {
            "model_id": GLINER_MODEL_ID,
            "text": text,
            "labels": labels or GLINER_LABELS,
            "threshold": GLINER_THRESHOLD,
        }
    )
    try:
        result = subprocess.run(
            [sys.executable, str(GLINER_WORKER_PATH)],
            input=payload,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise TaskExecutionError("Failed to start the GLiNER PII subprocess.") from exc
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if "No module named 'gliner'" in stderr:
            raise RuntimeImportError(
                "Install 'aibackends[pii]' to use the GLiNER PII backend."
            )
        raise TaskExecutionError(f"GLiNER PII inference failed: {stderr or 'unknown error'}")
    if not result.stdout.strip():
        return []
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise TaskExecutionError("GLiNER PII returned invalid JSON.")
    return parsed if isinstance(parsed, list) else []


def _gliner_entities(text: str, *, labels: list[str] | None = None) -> list[PIIEntity]:
    raw_entities = _predict_gliner_raw_entities(text, labels=labels)
    entities: list[PIIEntity] = []
    for item in raw_entities:
        start = int(item.get("start", 0))
        end = int(item.get("end", start))
        entities.append(
            PIIEntity(
                entity_type=str(item.get("label", "PII")).upper().replace(" ", "_"),
                text=text[start:end],
                start=start,
                end=end,
                replacement="",
            )
        )
    return entities


def _privacy_filter_entities(text: str) -> list[PIIEntity]:
    if sys.version_info >= (3, 13):
        raise TaskExecutionError(
            "The openai/privacy-filter backend is not supported in this Python runtime."
        )
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeImportError(
            "Install 'aibackends[pii]' to use the openai/privacy-filter backend."
        ) from exc

    detector = pipeline(
        "token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
    )
    entities: list[PIIEntity] = []
    for item in detector(text):
        start = int(item.get("start", 0))
        end = int(item.get("end", start))
        entities.append(
            PIIEntity(
                entity_type=str(item.get("entity_group", "PII")).upper(),
                text=text[start:end],
                start=start,
                end=end,
                replacement="",
            )
        )
    return entities


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

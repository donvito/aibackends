from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.core.registry import PIIBackendSpec
from aibackends.schemas.pii import PIIEntity

GLINER_MODEL_ID = "nvidia/gliner-pii"
GLINER_LABELS = ("email", "phone_number", "user_name")
GLINER_THRESHOLD = 0.5
GLINER_WORKER_PATH = Path(__file__).with_name("worker.py")


def detect_entities(
    spec: PIIBackendSpec,
    text: str,
    labels: list[str] | None,
) -> list[PIIEntity]:
    raw_entities = _predict_raw_entities(spec, text, labels=labels)
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


def _predict_raw_entities(
    spec: PIIBackendSpec,
    text: str,
    *,
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    payload = json.dumps(
        {
            "model_id": spec.model_id,
            "text": text,
            "labels": labels or list(spec.default_labels),
            "threshold": spec.threshold,
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
    except json.JSONDecodeError as exc:
        raise TaskExecutionError("GLiNER PII returned invalid JSON.") from exc
    return parsed if isinstance(parsed, list) else []


PII_BACKEND_SPEC = PIIBackendSpec(
    name="gliner",
    detect=detect_entities,
    model_id=GLINER_MODEL_ID,
    default_labels=GLINER_LABELS,
    threshold=GLINER_THRESHOLD,
    supports_custom_labels=True,
    metadata={"worker_path": str(GLINER_WORKER_PATH)},
)

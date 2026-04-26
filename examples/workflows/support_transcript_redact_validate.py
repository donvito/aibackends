"""Support transcript PII workflow.

Implements the five-step workflow:
1. ingest   -> read transcript and detect language
2. detect   -> run local NER with the GLiNER backend
3. classify -> normalize entity labels and keep confidence scores
4. redact   -> replace spans and keep a redaction map
5. validate -> validate the final payload against RedactedText

Uses ``examples/data/support_transcript.txt`` and stays fully local.

Requires:
    pip install 'aibackends[pii]'
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from aibackends.backends.pii import get_pii_backend
from aibackends.backends.pii.gliner import GLINER_WORKER_PATH
from aibackends.core.exceptions import AIBackendsError, RuntimeImportError, TaskExecutionError
from aibackends.core.types import AIBackendsModel
from aibackends.schemas.pii import PIIEntity, RedactedText
from aibackends.steps._base import BaseStep, StepContext
from aibackends.tasks._utils import load_text_input
from aibackends.workflows import Pipeline

SUPPORT_PII_LABELS = [
    "person_name",
    "email",
    "case_number",
    "transaction_reference",
]

ENGLISH_HINTS = {
    "the",
    "and",
    "can",
    "customer",
    "email",
    "help",
    "payment",
    "support",
    "thanks",
    "today",
    "vendor",
    "you",
}


def _coerce_payload(payload: Any) -> dict[str, Any]:
    return payload.copy() if isinstance(payload, dict) else {"input": payload}


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
    return normalized.upper() or "PII"


def _detect_language(text: str) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return "unknown"

    english_hits = sum(1 for word in words[:200] if word in ENGLISH_HINTS)
    ascii_ratio = sum(1 for char in text if char.isascii()) / max(1, len(text))
    if ascii_ratio > 0.95 and english_hits >= 8:
        return "en"
    return "unknown"


def _predict_gliner_entities(text: str, labels: list[str] | None) -> list[dict[str, Any]]:
    backend_spec = get_pii_backend("gliner")
    payload = json.dumps(
        {
            "model_id": backend_spec.model_id,
            "text": text,
            "labels": labels or list(backend_spec.default_labels),
            "threshold": backend_spec.threshold,
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
            raise RuntimeImportError("Install 'aibackends[pii]' to use the GLiNER backend.")
        raise TaskExecutionError(f"GLiNER PII inference failed: {stderr or 'unknown error'}")

    if not result.stdout.strip():
        return []

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise TaskExecutionError("GLiNER PII returned invalid JSON.") from exc

    return parsed if isinstance(parsed, list) else []


class ClassifiedEntity(AIBackendsModel):
    label: str
    text: str
    start: int
    end: int
    confidence: float


class TranscriptIngestor(BaseStep):
    name = "ingest"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        source = data.get("input", data.get("path"))
        if source is None:
            raise TaskExecutionError("Ingest step requires an input path.")

        path = Path(str(source)).expanduser()
        text = load_text_input(path)
        if not text.strip():
            raise TaskExecutionError("Transcript is empty.")

        data["path"] = str(path)
        data["text"] = text
        data["language"] = _detect_language(text)
        return data


class GLiNEREntityDetector(BaseStep):
    name = "detect"

    def __init__(self, *, labels: list[str] | None = None) -> None:
        self.labels = labels

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        text = str(data.get("text", ""))
        data["pii_backend"] = "gliner"
        data["detected_entities"] = _predict_gliner_entities(text, self.labels)
        return data


class EntityClassifier(BaseStep):
    name = "classify"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        text = str(data.get("text", ""))
        raw_entities = data.get("detected_entities", [])
        if not isinstance(raw_entities, list):
            raise TaskExecutionError("Detect step must provide a list of entities.")

        classified_entities: list[ClassifiedEntity] = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue

            start = int(item.get("start", 0))
            end = int(item.get("end", start))
            if start < 0 or end <= start or end > len(text):
                continue

            classified_entities.append(
                ClassifiedEntity(
                    label=_normalize_label(str(item.get("label", "PII"))),
                    text=text[start:end],
                    start=start,
                    end=end,
                    confidence=float(item.get("score", 0.0)),
                )
            )

        classified_entities.sort(
            key=lambda entity: (entity.start, -entity.confidence, entity.label)
        )
        data["classified_entities"] = classified_entities
        return data


class EntityRedactor(BaseStep):
    name = "redact"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        text = str(data.get("text", ""))
        raw_entities = data.get("classified_entities", [])
        if not isinstance(raw_entities, list):
            raise TaskExecutionError("Classify step must provide a list of classified entities.")

        entities = [item for item in raw_entities if isinstance(item, ClassifiedEntity)]
        parts: list[str] = []
        cursor = 0
        redaction_map: dict[str, str] = {}
        entities_found: list[PIIEntity] = []

        for index, entity in enumerate(entities, start=1):
            if entity.start < cursor:
                continue

            replacement = f"[{entity.label}_{index}]"
            parts.append(text[cursor : entity.start])
            parts.append(replacement)
            cursor = entity.end
            redaction_map[entity.text] = replacement
            entities_found.append(
                PIIEntity(
                    entity_type=entity.label,
                    text=entity.text,
                    start=entity.start,
                    end=entity.end,
                    replacement=replacement,
                )
            )

        parts.append(text[cursor:])
        redacted_text = "".join(parts)
        data["redacted_text"] = redacted_text
        data["redaction_payload"] = {
            "original_text": text,
            "redacted_text": redacted_text,
            "entities_found": entities_found,
            "redaction_map": redaction_map,
            "backend_used": str(data.get("pii_backend", "gliner")),
        }
        return data


class RedactedTextValidator(BaseStep):
    name = "validate"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        try:
            data["validated_redaction"] = RedactedText.model_validate(data["redaction_payload"])
        except ValidationError as exc:
            raise TaskExecutionError("RedactedText validation failed.") from exc
        return data


class SupportTranscriptRedactionWorkflow(Pipeline):
    steps = [
        TranscriptIngestor(),
        GLiNEREntityDetector(labels=SUPPORT_PII_LABELS),
        EntityClassifier(),
        EntityRedactor(),
        RedactedTextValidator(),
    ]


def main() -> None:
    try:
        transcript_path = Path(__file__).parent.parent / "data" / "support_transcript.txt"
        workflow = SupportTranscriptRedactionWorkflow()
        result = workflow.run(transcript_path)

        classified_entities = result.get("classified_entities")
        validated_redaction = result.get("validated_redaction")
        if not isinstance(classified_entities, list) or not isinstance(
            validated_redaction, RedactedText
        ):
            raise TaskExecutionError("Workflow returned an unexpected result shape.")

        print(f"Language detected: {result.get('language', 'unknown')}", flush=True)
        print(f"GLiNER entities classified: {len(classified_entities)}", flush=True)
        for entity in classified_entities:
            if isinstance(entity, ClassifiedEntity):
                print(
                    f"- {entity.label:<24} {entity.confidence:.3f}  {entity.text}",
                    flush=True,
                )

        print("\nValidated RedactedText:", flush=True)
        print(validated_redaction.model_dump_json(indent=2), flush=True)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

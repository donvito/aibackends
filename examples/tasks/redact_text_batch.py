"""Sequential PII redaction with one GLiNER model load.

Loads the GLiNER PII model once, then redacts multiple text files from
``examples/data/batch`` one at a time. This keeps the model warm so later
documents avoid paying the full initial load cost again.

Requires:
    pip install 'aibackends[pii]'
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from aibackends.backends.pii import get_pii_backend
from aibackends.core.exceptions import AIBackendsError, RuntimeImportError, TaskExecutionError
from aibackends.schemas.pii import PIIEntity, RedactedText

PII_LABELS = [
    "person_name",
    "email",
    "phone_number",
    "address",
]


def _load_gliner_model(model_id: str) -> Any:
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeImportError("Install 'aibackends[pii]' to run this example.") from exc

    return GLiNER.from_pretrained(model_id)


def _predict_entities(
    model: Any,
    text: str,
    *,
    labels: list[str],
    threshold: float,
) -> list[PIIEntity]:
    raw_entities = model.predict_entities(text, labels, threshold=threshold)
    entities: list[PIIEntity] = []
    for item in raw_entities:
        start = int(item.get("start", 0))
        end = int(item.get("end", start))
        if start < 0 or end <= start or end > len(text):
            continue

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


def _build_redaction(text: str, entities: list[PIIEntity], *, backend_name: str) -> RedactedText:
    parts: list[str] = []
    cursor = 0
    redaction_map: dict[str, str] = {}
    normalized_entities: list[PIIEntity] = []

    for index, entity in enumerate(sorted(entities, key=lambda item: item.start), start=1):
        if entity.start < cursor:
            continue

        replacement = f"[{entity.entity_type}_{index}]"
        parts.append(text[cursor : entity.start])
        parts.append(replacement)
        cursor = entity.end
        redaction_map[entity.text] = replacement
        normalized_entities.append(entity.model_copy(update={"replacement": replacement}))

    parts.append(text[cursor:])
    return RedactedText(
        original_text=text,
        redacted_text="".join(parts),
        entities_found=normalized_entities,
        redaction_map=redaction_map,
        backend_used=backend_name,
    )


def _load_batch_inputs() -> list[Path]:
    data_dir = Path(__file__).parent.parent / "data" / "batch"
    input_paths = sorted(data_dir.glob("pii_note_*.txt"))
    if not input_paths:
        raise TaskExecutionError(f"No PII batch inputs found in {data_dir}.")
    return input_paths


def main() -> None:
    try:
        backend_spec = get_pii_backend("gliner")
        if backend_spec.model_id is None:
            raise TaskExecutionError("The GLiNER backend is missing a model_id.")

        input_paths = _load_batch_inputs()
        print(f"Found {len(input_paths)} inputs in {input_paths[0].parent}", flush=True)
        for path in input_paths:
            print(f"- {path.name}", flush=True)

        print("\nLoading GLiNER once...", flush=True)
        load_started = time.perf_counter()
        model = _load_gliner_model(backend_spec.model_id)
        load_elapsed_ms = int((time.perf_counter() - load_started) * 1000)
        print(f"Model loaded in {load_elapsed_ms} ms", flush=True)

        print("\nRedacting inputs sequentially...", flush=True)
        for path in input_paths:
            text = path.read_text(encoding="utf-8")
            started = time.perf_counter()
            entities = _predict_entities(
                model,
                text,
                labels=PII_LABELS,
                threshold=backend_spec.threshold or 0.5,
            )
            result = _build_redaction(text, entities, backend_name=backend_spec.name)
            elapsed_ms = int((time.perf_counter() - started) * 1000)

            print(
                f"\n{path.name}: {len(result.entities_found)} entities in {elapsed_ms} ms",
                flush=True,
            )
            print(result.redacted_text, flush=True)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

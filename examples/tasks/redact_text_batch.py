"""Sequential PII redaction with one GLiNER model load.

Loads the GLiNER PII model once, then redacts multiple text files from
``examples/data/batch`` one at a time. The first ``backend.redact(...)`` call
warms the in-process model cache so later documents skip the heavy load cost.

Requires:
    pip install 'aibackends[pii]'
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from aibackends.backends.pii import get_pii_backend
from aibackends.core.exceptions import AIBackendsError, TaskExecutionError

PII_LABELS = [
    "person_name",
    "email",
    "phone_number",
    "address",
]


def _load_batch_inputs() -> list[Path]:
    data_dir = Path(__file__).parent.parent / "data" / "batch"
    input_paths = sorted(data_dir.glob("pii_note_*.txt"))
    if not input_paths:
        raise TaskExecutionError(f"No PII batch inputs found in {data_dir}.")
    return input_paths


def main() -> None:
    try:
        backend = get_pii_backend("gliner")

        input_paths = _load_batch_inputs()
        print(f"Found {len(input_paths)} inputs in {input_paths[0].parent}", flush=True)
        for path in input_paths:
            print(f"- {path.name}", flush=True)

        print("\nLoading GLiNER once...", flush=True)
        load_started = time.perf_counter()
        backend.load()
        load_elapsed_ms = int((time.perf_counter() - load_started) * 1000)
        print(f"Model loaded in {load_elapsed_ms} ms", flush=True)

        print("\nRedacting inputs sequentially...", flush=True)
        for path in input_paths:
            text = path.read_text(encoding="utf-8")
            started = time.perf_counter()
            result = backend.redact(text, labels=PII_LABELS)
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

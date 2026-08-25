"""Attach clinical context such as negation and dosage form to extracted spans.

Run from the repository root:
    python3 -m examples.gliner25.span_attributes --model small --device cpu

Requires:
    pip install -e '.[gliner2]'
"""

from __future__ import annotations

import argparse
import time

from aibackends.tasks import extract_schema

from .common import add_model_arguments, assert_source_spans, load_backend, print_result

TEXT = (
    "Patient reports a severe headache but denies chest pain. "
    "She started one 400 mg ibuprofen tablet every six hours and has no nausea."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    backend, model_id, device = load_backend(args.model, args.device)

    schema = (
        backend.create_schema(model=args.model, device=device)
        .entities(
            {
                "symptom": "Symptoms or clinical findings mentioned by the patient",
                "medication": "Medication or drug names",
                "dosage": "Medication dose amounts",
            }
        )
        .entity_attributes(
            {
                "negation_status": backend.create_attribute_group(
                    ["present", "negated"],
                    applies_to=["symptom"],
                    qualify_labels=True,
                ),
                "dosage_form": backend.create_attribute_group(
                    ["tablet", "capsule", "liquid", "injection", "unspecified"],
                    applies_to=["medication"],
                    qualify_labels=True,
                ),
            }
        )
    )
    result = extract_schema(
        TEXT,
        schema,
        backend=backend.name,
        model=args.model,
        device=device,
        include_spans=True,
        include_confidence=True,
    )
    span_count = assert_source_spans(TEXT, result)

    print(f"model: {model_id}")
    print(f"device: {device}")
    print(f"load + extraction: {time.perf_counter() - started:.2f}s")
    print(f"verified spans: {span_count}")
    print_result(result)


if __name__ == "__main__":
    main()

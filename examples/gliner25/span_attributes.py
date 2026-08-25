"""Clinical extraction with span attributes decoded in the same forward pass.

Each extracted span comes back qualified: symptoms carry negation status and
medications carry an active/discontinued status, without a second
classification pass over the extracted spans.

Requires:
    pip install 'aibackends[extraction]'
"""

from __future__ import annotations

import argparse

from aibackends.tasks import extract_entities

NOTE = (
    "Patient denies chest pain but reports severe headache and intermittent "
    "dizziness. Prescribed 400mg ibuprofen twice daily for the headache. "
    "Aspirin was discontinued last month due to a mild allergy."
)

LABELS = {
    "symptom": "Symptoms or complaints mentioned for the patient",
    "medication": "Names of drugs or pharmaceutical substances",
    "dosage": "Dose amounts such as 400mg or 2 tablets",
}

ATTRIBUTES = {
    "negation": {
        "labels": ["present", "denied by patient"],
        "applies_to": ["symptom"],
    },
    "status": {
        "labels": ["currently prescribed", "discontinued"],
        "applies_to": ["medication"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="base", help="small, base, multi, or a HF repo id.")
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, cuda:<index>, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = extract_entities(
        NOTE,
        labels=LABELS,
        attributes=ATTRIBUTES,
        model=args.model,
        device=args.device,
        threshold=0.4,
    )

    for entity in result.entities:
        qualifiers = ", ".join(
            f"{group}={attribute.label}" for group, attribute in entity.attributes.items()
        )
        suffix = f"  ({qualifiers})" if qualifiers else ""
        print(f"[{entity.label}] {entity.text}{suffix}")


if __name__ == "__main__":
    main()

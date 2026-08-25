"""Detect PII entities with GLiNER2.5 and redact them using character offsets.

Every extracted span carries half-open character offsets into the source text
(text[start:end] == span text), so redaction can happen at the source without
re-searching for the matched string.

Requires:
    pip install 'aibackends[extraction]'
"""

from __future__ import annotations

import argparse

from aibackends.tasks import extract_entities

PII_LABELS = {
    "person_name": "Full names of people",
    "email_address": "Email addresses",
    "phone_number": "Phone numbers in any format",
    "postal_address": "Complete postal or street addresses",
    "credit_card_number": "Payment card numbers",
}

TEXT = (
    "Please update the billing contact to Maria Gonzales, reachable at "
    "maria.gonzales@example.com or +1 (404) 555-0182. Ship the replacement "
    "card ending 4111 1111 1111 1111 to 4800 Lakeside Commons Drive, Suite "
    "1200, Atlanta, Georgia 30339."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small", help="small, base, multi, or a HF repo id.")
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, cuda:<index>, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = extract_entities(
        TEXT,
        labels=PII_LABELS,
        model=args.model,
        device=args.device,
        threshold=0.4,
    )

    print("Detected PII entities")
    print(result.model_dump_json(indent=2))

    redacted = TEXT
    for entity in sorted(result.entities, key=lambda item: item.start or 0, reverse=True):
        if entity.start is None or entity.end is None:
            continue
        redacted = (
            redacted[: entity.start]
            + f"[{entity.label.upper()}]"
            + redacted[entity.end :]
        )

    print("\nRedacted text")
    print(redacted)


if __name__ == "__main__":
    main()

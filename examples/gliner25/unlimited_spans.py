"""Extract spans that exceed legacy span-width limits.

GLiNER2's span enumeration capped entities at roughly twelve words; anything
longer was structurally invisible. The GLiNER2.5 boundary architecture scores
where a span starts and ends, so a twenty-five-word quote costs the same to
locate as a two-word name.

Requires:
    pip install 'aibackends[extraction]'
"""

from __future__ import annotations

import argparse

from aibackends.tasks import extract_entities

TEXT = (
    'During the earnings call, CEO Amara Osei said "we expect revenue to '
    "grow by twenty percent next year driven by strong demand in our cloud "
    'division and continued expansion into Southeast Asian markets", before '
    "asking investors to send written questions to Meridian Cloud Holdings, "
    "Investor Relations, 4800 Lakeside Commons Drive, Suite 1200, Atlanta, "
    "Georgia 30339, United States."
)

LABELS = {
    "quote": "The complete quoted statement",
    "postal_address": "A complete postal address including street, city, and country",
    "person_name": "Full names of people",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="base", help="small, base, multi, or a HF repo id.")
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, cuda:<index>, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = extract_entities(
        TEXT,
        labels=LABELS,
        model=args.model,
        device=args.device,
        threshold=0.4,
    )

    for entity in result.entities:
        words = len(entity.text.split())
        confidence = f"{entity.confidence:.2f}" if entity.confidence is not None else "n/a"
        beyond = "  <- beyond GLiNER2's ~12-word span cap" if words > 12 else ""
        print(f"[{entity.label}] {words} words, confidence {confidence}{beyond}")
        print(f"  chars {entity.start}..{entity.end}: {entity.text}")
        if entity.start is not None and entity.end is not None:
            verified = TEXT[entity.start : entity.end] == entity.text
            print(f"  offset check: {verified}")
        print()


if __name__ == "__main__":
    main()

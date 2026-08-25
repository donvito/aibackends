"""Extract entities from a full contract with native long-document chunking.

The document is split into overlapping word chunks, extraction runs per chunk,
spans are remapped to character offsets in the original document, and
duplicates across overlaps are merged. Every span verifies directly against
the source text.

Requires:
    pip install 'aibackends[extraction]'
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aibackends.tasks import extract_entities

CONTRACT_PATH = Path(__file__).parent.parent / "data" / "sample_contract.txt"

LABELS = {
    "party": "Named companies or organizations that are parties to the agreement",
    "person_name": "Full names of individual people",
    "monetary_amount": "Money amounts such as USD 84,500",
    "duration": "Time periods such as twenty-four months or ninety days",
    "governing_law": "The governing law or jurisdiction clause",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="small", help="small, base, multi, or a HF repo id.")
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, cuda:<index>, or mps.")
    parser.add_argument("--chunk-size", type=int, default=384, help="Words per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Overlap in words.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    print(f"Document length: {len(contract)} characters, {len(contract.split())} words\n")

    result = extract_entities(
        CONTRACT_PATH,
        labels=LABELS,
        model=args.model,
        device=args.device,
        threshold=0.4,
        long_document=True,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    verified = 0
    for entity in result.entities:
        if entity.start is not None and entity.end is not None:
            assert contract[entity.start : entity.end] == entity.text
            verified += 1
        print(f"[{entity.label}] chars {entity.start}..{entity.end}: {entity.text}")

    print(
        f"\n{len(result.entities)} entities extracted; "
        f"{verified} span offsets verified against the source text."
    )


if __name__ == "__main__":
    main()

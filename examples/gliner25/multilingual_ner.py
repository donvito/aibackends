"""Zero-shot multilingual NER with gliner2.5-multi-v1.

The multilingual checkpoint extracts entities from non-English text with
English label names, using native batch inference across languages.

Requires:
    pip install 'aibackends[extraction]'
"""

from __future__ import annotations

import argparse

from aibackends.tasks import extract_entities_batch

TEXTS = [
    "La empresa Iberdrola anunció una inversión de 3.000 millones de euros "
    "en Valencia junto a su presidente Ignacio Galán.",
    "Die Lufthansa eröffnet ein neues Drehkreuz in München, wie "
    "Vorstandschef Carsten Spohr am Montag erklärte.",
    "L'entreprise TotalEnergies a signé un accord avec le gouvernement du "
    "Sénégal à Dakar, selon Patrick Pouyanné.",
]

LABELS = ["person", "organization", "location", "monetary amount"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="multi", help="small, base, multi, or a HF repo id.")
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, cuda:<index>, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    results = extract_entities_batch(
        TEXTS,
        labels=LABELS,
        model=args.model,
        device=args.device,
        threshold=0.4,
        batch_size=4,
    )

    for text, result in zip(TEXTS, results, strict=True):
        print(f"text: {text}")
        for entity in result.entities:
            print(f"  [{entity.label}] {entity.text}")
        print()


if __name__ == "__main__":
    main()

"""Run entities, classification, relations, and structured records in one pass.

Run from the repository root:
    python3 -m examples.gliner25.combined_schema --model small --device cpu

Requires:
    pip install -e '.[gliner2]'
"""

from __future__ import annotations

import argparse
import time

from aibackends.tasks import extract_schema

from .common import add_model_arguments, assert_source_spans, load_backend, print_result

TEXT = (
    "Apple CEO Tim Cook announced the iPhone 15 for $999 in Cupertino. "
    "Reviewers praised the camera and called the launch exciting."
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
        .entities(["person", "company", "product", "location"])
        .classification("sentiment", ["positive", "negative", "neutral"])
        .classification("document_type", ["product_news", "review", "opinion"])
        .relations(["works_for", "announced_by", "located_in"])
        .structure("product")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("feature", dtype="list")
        .field(
            "category",
            dtype="str",
            choices=["phone", "computer", "software", "service"],
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
    print(f"load + one-pass extraction: {time.perf_counter() - started:.2f}s")
    print(f"verified spans: {span_count}")
    print_result(result)


if __name__ == "__main__":
    main()

"""Build a typed, internally consistent knowledge graph with Joint IE.

Run from the repository root:
    python3 -m examples.gliner25.joint_information_extraction --model small --device cpu

Requires:
    pip install -e '.[gliner2]'
"""

from __future__ import annotations

import argparse
import time

from aibackends.tasks import extract_graph

from .common import (
    add_model_arguments,
    assert_source_spans,
    load_backend,
    print_result,
)

TEXT = (
    "Tim Cook leads Apple in Cupertino. "
    "Sundar Pichai runs Google in Mountain View. "
    "Alice Chen joined Acme Robotics in Paris."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--beam-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.beam_size < 1:
        raise SystemExit("--beam-size must be at least 1")

    started = time.perf_counter()
    backend, model_id, device = load_backend(args.model, args.device)

    schema = (
        backend.create_joint_schema(model=args.model, device=device)
        .entities(["person", "organization", "location"])
        .relation("works_for", "person", "organization", unique_head=True)
        .relation("located_in", "organization", "location", unique_head=True)
        .no_self_loops()
    )
    result = extract_graph(
        TEXT,
        schema,
        backend=backend.name,
        model=args.model,
        device=device,
        config=backend.create_joint_config(
            optimizer="beam",
            beam_size=args.beam_size,
        ),
    )
    span_count = assert_source_spans(TEXT, result)

    print(f"model: {model_id}")
    print(f"device: {device}")
    print(f"load + extraction: {time.perf_counter() - started:.2f}s")
    print(f"feasible graph: {result.feasible}; {span_count} source-verified entity spans")
    print_result(result)

    print("\nResolved graph edges")
    for relation in result.relations:
        head = result.entity(relation.head)
        tail = result.entity(relation.tail)
        print(
            f"{head.text} -{relation.type}-> {tail.text} "
            f"(confidence {relation.confidence:.2f})"
        )


if __name__ == "__main__":
    main()

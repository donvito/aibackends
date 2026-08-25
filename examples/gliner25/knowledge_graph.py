"""Build a consistent knowledge graph with GLiNER2.5 joint extraction.

Entities and relations are decoded together as one globally consistent graph:
every relation endpoint exists in the result, typed endpoints are enforced
(works_for goes person -> organization), and schema rules such as unique_head
and no self-loops hold by construction. Check `feasible` to distinguish "the
constraints could not be satisfied" from "nothing to extract".

Requires:
    pip install 'aibackends[extraction]'
"""

from __future__ import annotations

import argparse

from aibackends.tasks import extract_graph

TEXT = "Tim Cook leads Apple in Cupertino. Sundar Pichai runs Google in Mountain View."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="base", help="small, base, multi, or a HF repo id.")
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, cuda:<index>, or mps.")
    parser.add_argument("--beam-size", type=int, default=32, help="Beam width for decoding.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    graph = extract_graph(
        TEXT,
        entities=["person", "organization", "location"],
        relations=[
            {
                "name": "works_for",
                "head": "person",
                "tail": "organization",
                "unique_head": True,
            },
            {"name": "located_in", "head": "organization", "tail": "location"},
        ],
        model=args.model,
        device=args.device,
        beam_size=args.beam_size,
    )

    print(f"feasible: {graph.feasible}\n")
    print("Entities")
    for entity in graph.entities:
        confidence = f"{entity.confidence:.2f}" if entity.confidence is not None else "n/a"
        print(f"  {entity.id}: [{entity.type}] {entity.text} (conf {confidence})")

    print("\nRelations")
    for relation in graph.relations:
        confidence = (
            f"{relation.confidence:.2f}" if relation.confidence is not None else "n/a"
        )
        print(
            f"  {relation.head_text} -{relation.type}-> {relation.tail_text} "
            f"(conf {confidence})"
        )


if __name__ == "__main__":
    main()

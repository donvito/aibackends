"""Run the main GLiNER 2.5 information-extraction use cases locally.

The examples cover long-document contract review, structured records, span
attributes, constrained agent routing, joint entity/relation extraction, and
multilingual extraction.

Usage:
    python examples/tasks/gliner25_information_extraction.py
    python examples/tasks/gliner25_information_extraction.py --model base
    python examples/tasks/gliner25_information_extraction.py --use-case routing graph

Requires:
    pip install 'aibackends[information-extraction]'
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from typing import Any

from gliner2 import AttributeGroup, AutoExtractor
from gliner2.classification import Classifier, ClassificationSchema
from gliner2.classification import constraints as C
from gliner2.joint_ie import JointIE, JointIEConfig

MODEL_IDS = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}

CONTRACT_PREAMBLE = " ".join(
    [
        "This agreement contains commercial background, definitions, service levels, "
        "and reporting terms."
    ]
    * 8
)
CONTRACT_TEXT = (
    f"{CONTRACT_PREAMBLE} Northstar Analytics LLC signed this services agreement with "
    "Blue Harbor Bank on August 18, 2026. Either party may terminate this Agreement for "
    "convenience by giving the other party at least thirty (30) days' prior written notice "
    f"delivered by certified mail. {CONTRACT_PREAMBLE}"
)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _verified_entity_offsets(text: str, result: dict[str, Any]) -> int:
    verified = 0
    entity_groups = result.get("entities", {})
    if not isinstance(entity_groups, dict):
        return verified

    for entities in entity_groups.values():
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, dict) or "start" not in entity or "end" not in entity:
                continue
            extracted = text[entity["start"] : entity["end"]]
            if extracted != entity["text"]:
                raise RuntimeError(
                    f"Invalid source offset for {entity['text']!r}: got {extracted!r}"
                )
            verified += 1
    return verified


def review_contract(model: Any) -> dict[str, Any]:
    """Scan every chunk of a contract and preserve source-document offsets."""
    result = model.extract_entities_long(
        CONTRACT_TEXT,
        {
            "contract party": "Organizations that are parties to the agreement",
            "effective date": "The date the agreement starts",
            "termination clause": "The complete clause describing how the agreement can end",
        },
        chunk_size=64,
        chunk_overlap=20,
        include_spans=True,
        include_confidence=True,
    )
    verified = _verified_entity_offsets(CONTRACT_TEXT, result)
    return {
        "document_words": len(CONTRACT_TEXT.split()),
        "verified_global_offsets": verified,
        **result,
    }


def extract_invoice(model: Any) -> dict[str, Any]:
    """Turn an invoice sentence into a schema-shaped record."""
    return model.extract_json(
        (
            "Invoice INV-2048 from Acme Labs totals $1,240.00 and is due "
            "September 30, 2026."
        ),
        {
            "invoice": [
                "invoice_number::str",
                "vendor::str",
                "total::str",
                "due_date::str",
            ]
        },
    )


def analyze_product_feedback(model: Any) -> dict[str, Any]:
    """Attach sentiment to each extracted product mention in one pass."""
    text = "The Atlas camera is excellent, but its battery life is disappointing."
    schema = (
        model.create_schema()
        .entities(["product"])
        .entity_attributes(
            {
                "sentiment": AttributeGroup(
                    ["positive", "negative", "neutral"],
                    applies_to=["product"],
                    qualify_labels=True,
                )
            }
        )
    )
    result = model.extract(
        text,
        schema,
        include_spans=True,
        include_confidence=True,
    )
    verified = _verified_entity_offsets(text, result)
    return {"verified_offsets": verified, **result}


def route_agent_action(model: Any) -> dict[str, Any]:
    """Decode an agent route and its effects under hard compatibility rules."""
    classifier = Classifier(model)
    schema = (
        ClassificationSchema()
        .single("route", ["read", "write", "delete"])
        .multi(
            "effects",
            ["read_only", "create", "modify", "delete"],
            min_labels=1,
            max_labels=2,
        )
        .constrain(
            C.implies(("route", "delete"), ("effects", "delete")),
            C.implies(("route", "read"), ("effects", "read_only")),
            C.excludes(("route", "read"), ("effects", "delete")),
            C.excludes(("route", "read"), ("effects", "modify")),
        )
    )
    return classifier.classify(
        "Delete the temporary file from /tmp.",
        schema,
    ).to_dict()


def build_agent_memory_graph(model: Any) -> dict[str, Any]:
    """Extract a typed, internally consistent graph for agent memory."""
    joint = JointIE(model)
    schema = (
        joint.create_schema()
        .entities(["person", "organization", "location"])
        .relation("works_for", "person", "organization", unique_head=True)
        .relation("located_in", "organization", "location", unique_head=True)
        .no_self_loops()
    )
    result = joint.extract(
        "Maya Chen leads Northstar Analytics in Singapore.",
        schema,
        config=JointIEConfig(optimizer="beam", beam_size=16),
    )
    payload = result.to_dict()
    payload["_meta"] = {"feasible": result.feasible}
    return payload


def extract_multilingual_entities(model: Any) -> dict[str, Any]:
    """Extract the same schema from Spanish text; use the multi model for best quality."""
    text = (
        "María González trabaja para Banco del Sol en Madrid desde el 12 de marzo de 2024."
    )
    result = model.extract_entities(
        text,
        ["person", "organization", "location", "date"],
        include_spans=True,
        include_confidence=True,
    )
    verified = _verified_entity_offsets(text, result)
    return {"verified_offsets": verified, **result}


USE_CASES: dict[str, Callable[[Any], dict[str, Any]]] = {
    "contract": review_contract,
    "invoice": extract_invoice,
    "attributes": analyze_product_feedback,
    "routing": route_agent_action,
    "graph": build_agent_memory_graph,
    "multilingual": extract_multilingual_entities,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_IDS),
        default="small",
        help="small is fastest, base is the English default, multi is multilingual.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Inference device. auto prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--use-case",
        nargs="+",
        choices=("all", *USE_CASES),
        default=["all"],
        help="One or more use cases to run.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    model_id = MODEL_IDS[args.model]
    selected = list(USE_CASES) if "all" in args.use_case else args.use_case

    print(f"Loading {model_id} on {device}...", flush=True)
    started = time.perf_counter()
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    print(f"Loaded in {time.perf_counter() - started:.2f}s", flush=True)

    for name in selected:
        print(f"\n=== {name} ===")
        result = USE_CASES[name](model)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

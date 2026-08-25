"""Route agent actions with constrained classification.

Two tasks are decoded jointly under declared rules: the action intent and its
side effects. Constraints make contradictory outputs impossible, so no
downstream reconciliation code is needed. Without constraints, the tasks are
decoded independently and can disagree.

Requires:
    pip install 'aibackends[extraction]'
"""

from __future__ import annotations

import argparse

from aibackends.tasks import classify_text

TASKS = {
    "intent": {"labels": ["read", "write", "delete"]},
    "effects": {
        "labels": ["read_only", "create", "modify", "delete"],
        "multi_label": True,
        "min_labels": 1,
        "max_labels": 2,
    },
}

CONSTRAINTS = [
    {"kind": "implies", "when": ["intent", "delete"], "then": ["effects", "delete"]},
    {"kind": "implies", "when": ["intent", "read"], "then": ["effects", "read_only"]},
    {"kind": "excludes", "when": ["intent", "read"], "then": ["effects", "delete"]},
    {"kind": "excludes", "when": ["intent", "read"], "then": ["effects", "modify"]},
]

REQUESTS = [
    "Delete the temporary files from /tmp before the backup runs",
    "Preview the quarterly report without changing anything",
    "Append the new customer records to the ledger",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="base", help="small, base, multi, or a HF repo id.")
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, cuda:<index>, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for request in REQUESTS:
        unconstrained = classify_text(
            request,
            tasks=TASKS,
            model=args.model,
            device=args.device,
        )
        constrained = classify_text(
            request,
            tasks=TASKS,
            constraints=CONSTRAINTS,
            model=args.model,
            device=args.device,
        )
        print(f"request: {request}")
        print(
            "  unconstrained: "
            f"intent={unconstrained.value('intent')} "
            f"effects={unconstrained.values('effects')}"
        )
        print(
            "  constrained:   "
            f"intent={constrained.value('intent')} "
            f"effects={constrained.values('effects')} "
            f"feasible={constrained.feasible}"
        )
        print()


if __name__ == "__main__":
    main()

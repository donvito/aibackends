"""Code-language routing with the LFM2.5 prompt router.

Domain-specific routing: send each bug report to the right language expert
(a specialised model, a docs index, or an on-call channel). The lanes are
just language names -- the router works out which one a report belongs to
even when the language is never mentioned by name.

Requires:
    pip install 'aibackends[routing]'

Usage:
    python examples/routing/route_code_language.py
    python examples/routing/route_code_language.py --threshold 0.3 --device cpu
"""

from __future__ import annotations

import argparse
import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.schemas.routing import RoutingResult
from aibackends.tasks import route_prompt

ROUTES = [
    "Python",
    "JavaScript",
    "TypeScript",
    "Go",
    "Rust",
    "Java",
    "C++",
    "PHP",
    "Ruby",
]

PROMPTS = [
    "np.einsum returns the wrong shape when broadcasting over the batch axis.",
    "The borrow checker rejects the lifetime in my iterator adapter.",
    "My .ts file fails to compile: the interface does not satisfy the constraint.",
    "My Laravel migration fails with a foreign key constraint error.",
    "go func leaks goroutines when the context is cancelled before the channel send.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Drop lanes scoring below this value (0-1, default: keep all).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, gpu, cuda, cuda:<index>, or mps (default: cpu).",
    )
    return parser.parse_args()


def print_result(result: RoutingResult) -> None:
    print(f"prompt: {result.text}")
    if not result.scores:
        print("  (no lane met the threshold)")
        print()
        return
    for index, score in enumerate(result.scores):
        marker = "->" if index == 0 else "  "
        bar = "#" * max(1, round(score.score * 24))
        print(f"  {marker} {score.route:<12} {score.score:6.1%}  {bar}")
    print()


def main() -> None:
    args = parse_args()
    print(f"routes: {ROUTES}")
    print()
    try:
        for prompt in PROMPTS:
            result = route_prompt(
                prompt,
                ROUTES,
                device=args.device,
                threshold=args.threshold,
            )
            print_result(result)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

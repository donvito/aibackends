"""Adding a routing category on the fly with the LFM2.5 prompt router.

Lanes are free text with no training step, so extending the taxonomy is just
appending a string. This demo routes a soccer question against the standard
device-assistant lanes, then adds a "Soccer agent" lane and routes the same
question again -- watch it leave the generic lane and claim its own agent.

Requires:
    pip install 'aibackends[routing]'

Usage:
    python examples/routing/route_custom_category.py
    python examples/routing/route_custom_category.py --category "Soccer agent" \
        --prompt "Who won the 2026 FIFA World Cup?"
"""

from __future__ import annotations

import argparse
import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import route_prompt

BASE_ROUTES = [
    "Simple function call",
    "Simple tool use",
    "Complex multi-step agentic task",
    "Quick factual question",
    "Casual conversation",
    "Creative writing",
    "Translation",
    "Needs a bigger model",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category",
        default="Soccer agent",
        help="Custom lane to add on the fly (default: 'Soccer agent').",
    )
    parser.add_argument(
        "--prompt",
        default="Who won the 2026 FIFA World Cup?",
        help="Prompt that the new lane should claim.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="cpu, gpu, cuda, cuda:<index>, or mps (default: cpu).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"prompt: {args.prompt}")
    print()
    try:
        before = route_prompt(args.prompt, BASE_ROUTES, device=args.device)
        after = route_prompt(
            args.prompt,
            [*BASE_ROUTES, args.category],
            device=args.device,
        )
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        f"without {args.category!r}: "
        f"{before.best_route!r} ({before.scores[0].score:.1%})"
    )
    print(
        f"with    {args.category!r}: "
        f"{after.best_route!r} ({after.scores[0].score:.1%})"
    )


if __name__ == "__main__":
    main()

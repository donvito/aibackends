"""Zero-shot prompt routing with LiquidAI LFM2.5-Encoder-350M-Prompt-Router.

The smallest possible routing example: pick a few free-text lanes, route a
prompt, print the ranked scores. Lanes are ordinary strings supplied at call
time -- no fixed taxonomy and no per-label training. The 350M bidirectional
encoder reads the whole prompt in one forward pass and scores it against
every lane at once, fast enough to run on CPU.

Scenario-specific demos live next to this file:

- route_device_assistant.py  device assistant orchestration
- route_code_language.py     code-language routing
- route_support_intent.py    support-ticket intent classification
- route_custom_category.py   adding a category on the fly

Requires:
    pip install 'aibackends[routing]'

Usage:
    python examples/routing/route_prompt.py
    python examples/routing/route_prompt.py --threshold 0.3 --device cpu
"""

from __future__ import annotations

import argparse
import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.schemas.routing import RoutingResult
from aibackends.tasks import route_prompt

ROUTES = ["coding", "sales", "creative writing", "general knowledge"]

PROMPTS = [
    "Can you help me debug a failing Python unit test?",
    "Draft a follow-up email for the enterprise deal we discussed.",
    "Write a haiku about the sea.",
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
        print(f"  {marker} {score.route:<20} {score.score:6.1%}  {bar}")
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

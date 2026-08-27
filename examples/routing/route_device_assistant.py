"""Device-assistant orchestration with the LFM2.5 prompt router.

The scenario from the Liquid AI demo: an on-device assistant separates cheap
requests (a weather lookup, a timer) from work that needs a real agent or a
bigger model (planning and booking a trip). The router reads each prompt in
one CPU-friendly forward pass and picks the lane, so simple traffic never
wakes up the expensive machinery.

Requires:
    pip install 'aibackends[routing]'

Usage:
    python examples/routing/route_device_assistant.py
    python examples/routing/route_device_assistant.py --threshold 0.3 --device cpu
"""

from __future__ import annotations

import argparse
import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.schemas.routing import RoutingResult
from aibackends.tasks import route_prompt

ROUTES = [
    "Simple function call",
    "Simple tool use",
    "Complex multi-step agentic task",
    "Quick factual question",
    "Casual conversation",
    "Creative writing",
    "Translation",
    "Needs a bigger model",
]

PROMPTS = [
    "What's the temperature in San Francisco?",
    "Set a timer for 10 minutes.",
    "Plan and book a three-day trip to Tokyo with hotels under $200 a night.",
    "Translate 'good morning, see you tonight' to Japanese.",
    "How's your day going?",
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
        print(f"  {marker} {score.route:<34} {score.score:6.1%}  {bar}")
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

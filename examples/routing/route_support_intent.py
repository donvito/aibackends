"""Support-ticket intent classification with the LFM2.5 prompt router.

Sort incoming messages by intent at runtime without training a classifier.
The lanes are plain descriptions of your queues, so changing the taxonomy is
just editing a list. The encoder is multilingual (15 languages), so the same
English lanes catch tickets written in Spanish or Japanese.

Requires:
    pip install 'aibackends[routing]'

Usage:
    python examples/routing/route_support_intent.py
    python examples/routing/route_support_intent.py --threshold 0.3 --device cpu
"""

from __future__ import annotations

import argparse
import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.schemas.routing import RoutingResult
from aibackends.tasks import route_prompt

ROUTES = [
    "billing question",
    "bug report",
    "feature request",
    "account access problem",
    "off-topic",
]

PROMPTS = [
    "I was charged twice for my subscription this month.",
    "The export button crashes the app on Safari.",
    "Could you add dark mode to the dashboard?",
    "No puedo iniciar sesion, la pagina dice que mi cuenta esta bloqueada.",
    "ダークモードを追加してもらえますか？",
    "What's a good pizza place nearby?",
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
        print(f"  {marker} {score.route:<24} {score.score:6.1%}  {bar}")
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

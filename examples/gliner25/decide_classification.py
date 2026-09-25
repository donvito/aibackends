"""Operational decisions with GLiNER2.5-Decide.

GLiNER2.5-Decide is a 340M English classifier tuned for intent, routing,
sentiment, priority, policy, and multi-label tags. Several heads are scored in
one forward pass, and a task can carry a question (`prompt`), label
descriptions, or an ordinal scale.

Requires:
    pip install 'aibackends[extraction]'
"""

from __future__ import annotations

import argparse

from aibackends.tasks import classify_text, classify_texts

SUPPORT_INTENTS = [
    "order_status",
    "refund_request",
    "cancel_subscription",
    "update_payment",
    "login_problem",
    "shipping_delay",
    "bug_report",
    "speak_to_human",
    "other",
]
POLICY = ["allow", "personal_data", "harassment", "scam", "violence", "spam"]

INBOX = [
    "Your mailbox is almost full. Click here in the next hour or we will delete every message.",
    "This is the third time I have explained the same missing refund. Get me a person.",
    "I was double charged this morning, please refund one of the payments.",
    "Tracking for my order hasn't moved since Monday.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="decide",
        help="decide, decide-1b, multi-decide, or a HF repo id.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, cuda:<index>, or mps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = {"model": args.model, "device": args.device}

    review = (
        "Battery dies before lunch, but the keyboard and the screen are the best "
        "I have used on a laptop."
    )
    result = classify_text(
        review,
        tasks={
            "sentiment": {"labels": ["positive", "negative", "mixed", "neutral"]},
            "aspects": {
                "labels": ["battery", "keyboard", "screen", "camera", "price", "support"],
                "multi_label": True,
                "cls_threshold": 0.4,
            },
        },
        **options,
    )
    print("input:", review)
    print("  sentiment:", result.value("sentiment"))
    print("  aspects:  ", result.values("aspects"))
    print()

    pin_text = "Please reset the card PIN. The new one never arrived and the old one is locked."
    pin = classify_text(
        pin_text,
        tasks={
            "intent": {
                "labels": {
                    "card_pin_change": "The customer wants a new PIN or the current PIN replaced",
                    "card_lost": "The physical card is missing",
                    "balance_inquiry": "The customer wants the current balance",
                },
            },
        },
        **options,
    )
    print("input:", pin_text)
    print("  intent (labels with descriptions):", pin.value("intent"))
    print()

    book_review = (
        "Gave up after 40 pages. Flat characters and a plot you can see coming from the cover."
    )
    rating = classify_text(
        book_review,
        tasks={"rating": {"labels": [str(i) for i in range(11)], "ordinal": True}},
        **options,
    )
    print("input:", book_review)
    print("  ordinal rating (0-10):", rating.value("rating"))
    print()

    policies = classify_texts(INBOX, tasks={"policy": POLICY}, **options)
    allowed = [
        message
        for message, policy in zip(INBOX, policies, strict=True)
        if policy.value("policy") == "allow"
    ]
    routes = classify_texts(
        allowed,
        tasks={
            "handoff": {
                "labels": ["yes", "no"],
                "prompt": "Should this conversation be handed off to a human agent?",
            },
            "intent": SUPPORT_INTENTS,
            "urgency": ["low", "normal", "high", "critical"],
        },
        **options,
    )
    decisions = dict(zip(allowed, routes, strict=True))
    for message, policy in zip(INBOX, policies, strict=True):
        route = decisions.get(message)
        if route is None:
            action = f"block ({policy.value('policy')})"
        elif route.value("handoff") == "yes" or route.value("intent") == "speak_to_human":
            action = f"human_queue urgency={route.value('urgency')}"
        else:
            action = f"workflow:{route.value('intent')} urgency={route.value('urgency')}"
        print(f"{message[:60]:<62} -> {action}")


if __name__ == "__main__":
    main()

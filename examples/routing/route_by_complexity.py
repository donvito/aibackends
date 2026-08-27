"""Route prompts to the right model tier based on complexity.

The 350M LFM2.5 prompt router acts as a traffic controller in front of a
fleet of models: cheap local models absorb the high-volume easy traffic and
frontier cloud models are reserved for the prompts that actually need them.

Tiers in this demo:

- quick factual / small talk  -> LFM2.5-2.6B, executed locally via llama.cpp
- straightforward coding      -> Qwen3.8-27B, executed locally via llama.cpp
                                 (opt-in with --run-local-tiers, ~15 GB GGUF)
- complex multi-step work     -> GPT-5.6 Sol, Claude Opus 5 / Fable 5, or
                                 Grok 4.6 (decision printed)
- creative writing            -> GPT-5.6 Terra (decision printed)
- off-topic / low value       -> GPT-5.6 Luna, or decline (decision printed)

aibackends ships local runtimes only, so cloud lanes print the chosen
provider and model id -- that hand-off is where you would call the provider
SDK in a real deployment.

Requires:
    pip install 'aibackends[routing]'
    pip install 'aibackends[llamacpp]'  # to execute the local tiers

Usage:
    python examples/routing/route_by_complexity.py
    python examples/routing/route_by_complexity.py --skip-local
    python examples/routing/route_by_complexity.py --run-local-tiers --device cpu
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass

from aibackends.core.exceptions import AIBackendsError
from aibackends.core.registry import ModelRef
from aibackends.models import LFM25_2_6B, QWEN38_27B
from aibackends.tasks import route_prompt


@dataclass(frozen=True)
class ModelTarget:
    tier: str
    candidates: tuple[str, ...]  # "provider/model" strings, primary first
    local_ref: ModelRef | None = None
    heavy: bool = False  # only executed locally with --run-local-tiers


ROUTE_TARGETS: dict[str, ModelTarget] = {
    "quick factual question or casual small talk": ModelTarget(
        tier="local small model",
        candidates=("llamacpp/lfm2.5-2.6b",),
        local_ref=LFM25_2_6B,
    ),
    "straightforward coding or technical task": ModelTarget(
        tier="local open model",
        candidates=("llamacpp/qwen3.8-27b",),
        local_ref=QWEN38_27B,
        heavy=True,
    ),
    "complex multi-step agentic task or deep reasoning": ModelTarget(
        tier="frontier cloud model",
        candidates=(
            "openai/gpt-5.6-sol",
            "anthropic/claude-opus-5",
            "anthropic/claude-fable-5",
            "xai/grok-4.6",
        ),
    ),
    "creative writing": ModelTarget(
        tier="balanced cloud model",
        candidates=("openai/gpt-5.6-terra",),
    ),
    "off-topic or low-value request": ModelTarget(
        tier="cost control",
        candidates=("openai/gpt-5.6-luna",),
    ),
}

PROMPTS = [
    "What's the capital of Australia?",
    "Refactor this Python function to use dataclasses and add type hints.",
    "Plan and execute a migration of our billing system to a new provider, "
    "including a rollback strategy and a phased cutover across three regions.",
    "Write a short story about a lighthouse keeper who befriends a whale.",
    "What's a good pizza topping?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-local",
        action="store_true",
        help="Print routing decisions only; never execute local models.",
    )
    parser.add_argument(
        "--run-local-tiers",
        action="store_true",
        help="Also execute the heavy local tier (Qwen3.8-27B, ~15 GB GGUF).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for local models: auto, cpu, or gpu (default: auto).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=768,
        help="Generation budget for local model replies; LFM2.5 is a reasoning "
        "model and needs headroom to finish thinking (default: 768).",
    )
    return parser.parse_args()


# The worker keeps llama.cpp out of the router's process: torch (routing) and
# llama-cpp-python (generation) each bundle their own OpenMP runtime on macOS,
# and initializing both in a single process crashes.
_LOCAL_WORKER_SOURCE = """
import json
import sys

from aibackends import get_runtime
from aibackends.models import get_model_ref
from aibackends.runtimes import LLAMACPP

payload = json.loads(sys.stdin.read())
runtime = get_runtime(
    {
        "runtime": LLAMACPP,
        "model": get_model_ref(payload["model"]),
        "device": payload["device"],
        "max_tokens": payload["max_tokens"],
    }
)
response = runtime.complete(
    [
        # The cheap tier answers directly; without this nudge the reasoning
        # model spends the whole token budget thinking on open-ended asks.
        {"role": "system", "content": "Answer directly in at most three sentences."},
        {"role": "user", "content": payload["prompt"]},
    ]
)
content = response.content
if "</think>" in content:
    content = content.rsplit("</think>", 1)[-1]
sys.stdout.write(json.dumps({"content": content.strip()}))
"""


def run_local_model(ref: ModelRef, prompt: str, args: argparse.Namespace) -> str:
    payload = json.dumps(
        {
            "model": ref.name,
            "device": None if args.device == "auto" else args.device,
            "max_tokens": args.max_tokens,
            "prompt": prompt,
        }
    )
    worker = subprocess.run(
        [sys.executable, "-c", _LOCAL_WORKER_SOURCE],
        input=payload,
        capture_output=True,
        text=True,
    )
    if worker.returncode != 0:
        detail = worker.stderr.strip().splitlines()
        raise AIBackendsError(detail[-1] if detail else "Local model worker failed.")
    reply = json.loads(worker.stdout)["content"]
    return str(reply)


def handle(prompt: str, args: argparse.Namespace) -> None:
    result = route_prompt(prompt, list(ROUTE_TARGETS), device="cpu")
    best = result.best_route
    confidence = result.scores[0].score if result.scores else 0.0
    print(f"prompt: {prompt}")
    print(f"-> lane: {best!r} ({confidence:.1%})")

    target = ROUTE_TARGETS.get(best) if best else None
    if target is None:
        print("   no dispatch target configured for this lane")
        print()
        return

    print(f"   tier: {target.tier}")
    if len(target.candidates) == 1:
        print(f"   model: {target.candidates[0]}")
    else:
        print(f"   model candidates: {', '.join(target.candidates)}")

    if target.local_ref is None:
        print("   (cloud hand-off point: call the provider SDK here)")
    elif args.skip_local:
        print("   (local execution skipped via --skip-local)")
    elif target.heavy and not args.run_local_tiers:
        print("   (pass --run-local-tiers to actually run this ~15 GB local model)")
    else:
        print(f"   running locally via llama.cpp: {target.candidates[0]}")
        reply = run_local_model(target.local_ref, prompt, args)
        print(f"   reply: {reply}")
    print()


def main() -> None:
    args = parse_args()
    print("routing lanes:")
    for lane, target in ROUTE_TARGETS.items():
        print(f"  - {lane!r} -> {target.tier}")
    print()
    try:
        for prompt in PROMPTS:
            handle(prompt, args)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

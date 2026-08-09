"""Tool calling demo with LiquidAI LFM2.5-2.6B.

LFM2.5 supports native tool calling: the tool list is passed as a JSON array
in the system prompt, and the model replies with a Pythonic call such as
`[get_weather(city="Paris")]` between `<|tool_call_start|>` and
`<|tool_call_end|>` special tokens. The tool result is then returned to the
model with the `tool` role, and the model writes the final answer.
See https://huggingface.co/LiquidAI/LFM2.5-2.6B#tool-use for details.

Usage:
    python examples/tasks/tool_calling_lfm.py --runtime llamacpp --device cpu
    python examples/tasks/tool_calling_lfm.py --runtime transformers --device cpu
    python examples/tasks/tool_calling_lfm.py --runtime llamacpp --quantization Q8_0
"""

import argparse
import json
import sys

from aibackends import get_runtime
from aibackends.core.exceptions import AIBackendsError
from aibackends.core.tool_calls import clean_answer, extract_tool_calls
from aibackends.models import LFM25_2_6B
from aibackends.runtimes import LLAMACPP, TRANSFORMERS

TOOL_SCHEMAS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Paris"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "convert_currency",
        "description": "Convert an amount from one currency to another.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Amount to convert"},
                "from_currency": {"type": "string", "description": "ISO code, e.g. USD"},
                "to_currency": {"type": "string", "description": "ISO code, e.g. EUR"},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
    },
]


def get_weather(city: str) -> dict:
    """Stub weather lookup returning canned data."""
    return {
        "city": city,
        "temperature_c": 21,
        "condition": "partly cloudy",
        "humidity": "58%",
    }


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Stub currency conversion using fixed demo rates."""
    rates = {("USD", "EUR"): 0.86, ("EUR", "USD"): 1.16, ("USD", "GBP"): 0.74}
    rate = rates.get((from_currency.upper(), to_currency.upper()), 1.0)
    return {
        "amount": amount,
        "from_currency": from_currency.upper(),
        "to_currency": to_currency.upper(),
        "converted_amount": round(amount * rate, 2),
        "rate": rate,
    }


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "convert_currency": convert_currency,
}

def build_runtime_overrides(args: argparse.Namespace) -> dict:
    overrides = {
        "runtime": TRANSFORMERS if args.runtime == "transformers" else LLAMACPP,
        "model": LFM25_2_6B,
        "device": None if args.device == "auto" else args.device,
        "max_tokens": args.max_tokens,
        "extra_options": {"skip_special_tokens": False},
    }
    if args.quantization:
        overrides["quantization"] = args.quantization
    return overrides


def run_tool_loop(runtime, question: str) -> None:
    messages = [
        {
            "role": "system",
            "content": f"List of tools: {json.dumps(TOOL_SCHEMAS)}",
        },
        {"role": "user", "content": question},
    ]
    print(f"[user] {question}")

    print("[model] thinking and selecting a tool...")
    response = runtime.complete(messages)
    tool_calls = extract_tool_calls(response.content)
    if not tool_calls:
        print(f"[model] answered without tools: {clean_answer(response.content)}")
        return

    results = []
    for call in tool_calls:
        rendered_args = ", ".join(
            f"{key}={value!r}" for key, value in call.arguments.items()
        )
        print(f"[tool call] {call.name}({rendered_args})")
        function = TOOL_FUNCTIONS.get(call.name)
        if function is None:
            results.append({"error": f"Unknown tool: {call.name}"})
            continue
        result = function(**call.arguments)
        print(f"[tool result] {json.dumps(result)}")
        results.append(result)

    messages.append({"role": "assistant", "content": clean_answer(response.content)})
    messages.append({"role": "tool", "content": json.dumps(results)})

    print("[model] interpreting the tool result...")
    final = runtime.complete(messages)
    print(f"[final answer] {clean_answer(final.content)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LFM2.5-2.6B tool calling demo.")
    parser.add_argument(
        "--runtime",
        choices=["llamacpp", "transformers"],
        default="llamacpp",
        help="Local runtime to use (default: llamacpp).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Toggle CPU or GPU inference (default: auto-detect).",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="GGUF quantization for llamacpp, e.g. Q4_K_M (default), Q5_K_M, Q8_0.",
    )
    parser.add_argument(
        "--question",
        default="What is the weather in Paris right now?",
        help="Question that should trigger a tool call.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Generation budget per round (default: 1024).",
    )
    args = parser.parse_args()

    if args.runtime == "llamacpp":
        quantization_note = args.quantization or "default (Q4_K_M)"
    else:
        quantization_note = "n/a (GGUF only)"
    print(f"runtime={args.runtime} device={args.device} quantization={quantization_note}")
    try:
        runtime = get_runtime(build_runtime_overrides(args))
        run_tool_loop(runtime, args.question)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

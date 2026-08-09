"""Evaluate tool-call accuracy of a catalog model on a local runtime.

Runs a fixed set of labeled questions against the model with a tool list in
the system prompt (the LFM2.5 native tool-use format), parses the predicted
tool calls with `aibackends.core.tool_calls.extract_tool_calls`, and scores:

- Tool selection: the set of called tool names matches the expected set
  (including "no tool" cases, where the model must answer directly).
- Arguments: every expected call has a predicted call with the same name and
  normalized arguments (strings compared case-insensitively, numbers
  numerically).
- Exact match: both of the above.

Writes a markdown report to ``evals/reports/`` for committing to the repo.

Usage:
    python evals/eval_tool_calls.py --runtime llamacpp --device cpu
    python evals/eval_tool_calls.py --runtime transformers --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

# Keep eval output readable; heavy libraries are imported lazily by the runtime.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from aibackends import get_runtime
from aibackends.core.exceptions import AIBackendsError
from aibackends.core.tool_calls import ToolCall, extract_tool_calls

REPORTS_DIR = Path(__file__).parent / "reports"

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
    {
        "name": "get_time",
        "description": "Get the current local time in a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name, e.g. Tokyo"},
            },
            "required": ["city"],
        },
    },
]


@dataclass(frozen=True, slots=True)
class EvalCase:
    question: str
    expected_calls: tuple[ToolCall, ...] = ()


CASES = [
    EvalCase(
        question="What is the weather in Paris right now?",
        expected_calls=(ToolCall("get_weather", {"city": "Paris"}),),
    ),
    EvalCase(
        question="How warm is it in Tokyo today?",
        expected_calls=(ToolCall("get_weather", {"city": "Tokyo"}),),
    ),
    EvalCase(
        question="Is it raining in London at the moment?",
        expected_calls=(ToolCall("get_weather", {"city": "London"}),),
    ),
    EvalCase(
        question="Convert 100 US dollars to euros.",
        expected_calls=(
            ToolCall(
                "convert_currency",
                {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
            ),
        ),
    ),
    EvalCase(
        question="How much is 250 EUR in USD?",
        expected_calls=(
            ToolCall(
                "convert_currency",
                {"amount": 250, "from_currency": "EUR", "to_currency": "USD"},
            ),
        ),
    ),
    EvalCase(
        question="I have 75 British pounds. How many dollars is that?",
        expected_calls=(
            ToolCall(
                "convert_currency",
                {"amount": 75, "from_currency": "GBP", "to_currency": "USD"},
            ),
        ),
    ),
    EvalCase(
        question="What time is it in New York?",
        expected_calls=(ToolCall("get_time", {"city": "New York"}),),
    ),
    EvalCase(
        question="What's the weather in Rome, and how much is 50 USD in EUR?",
        expected_calls=(
            ToolCall("get_weather", {"city": "Rome"}),
            ToolCall(
                "convert_currency",
                {"amount": 50, "from_currency": "USD", "to_currency": "EUR"},
            ),
        ),
    ),
    EvalCase(question="What is the capital of France?"),
    EvalCase(question="Write a haiku about rain."),
]


@dataclass
class CaseResult:
    case: EvalCase
    predicted_calls: list[ToolCall]
    elapsed_ms: float
    selection_correct: bool = False
    arguments_correct: bool = False

    @property
    def exact_match(self) -> bool:
        return self.selection_correct and self.arguments_correct


def _normalise_value(value: object) -> object:
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def _normalise_call(call: ToolCall) -> tuple[str, tuple[tuple[str, object], ...]]:
    arguments = tuple(
        sorted((key, _normalise_value(value)) for key, value in call.arguments.items())
    )
    return (call.name, arguments)


def score_case(case: EvalCase, predicted: list[ToolCall]) -> tuple[bool, bool]:
    expected_names = sorted(call.name for call in case.expected_calls)
    predicted_names = sorted(call.name for call in predicted)
    selection_correct = expected_names == predicted_names

    expected_normalised = sorted(_normalise_call(call) for call in case.expected_calls)
    predicted_normalised = sorted(_normalise_call(call) for call in predicted)
    arguments_correct = expected_normalised == predicted_normalised
    return selection_correct, arguments_correct


def render_calls(calls: list[ToolCall] | tuple[ToolCall, ...]) -> str:
    if not calls:
        return "(no tool)"
    rendered = []
    for call in calls:
        arguments = ", ".join(f"{key}={value!r}" for key, value in call.arguments.items())
        rendered.append(f"{call.name}({arguments})")
    return "<br>".join(rendered)


def run_eval(args: argparse.Namespace) -> list[str]:
    overrides = {
        "runtime": args.runtime,
        "model": args.model,
        "device": None if args.device == "auto" else args.device,
        "max_tokens": args.max_tokens,
        "extra_options": {"skip_special_tokens": False},
    }
    if args.quantization:
        overrides["quantization"] = args.quantization

    runtime = get_runtime(overrides)
    system_prompt = f"List of tools: {json.dumps(TOOL_SCHEMAS)}"

    results: list[CaseResult] = []
    for index, case in enumerate(CASES, start=1):
        print(f"[{index}/{len(CASES)}] {case.question}", flush=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": case.question},
        ]
        started = time.perf_counter()
        response = runtime.complete(messages)
        elapsed_ms = (time.perf_counter() - started) * 1000
        predicted = extract_tool_calls(response.content)
        result = CaseResult(case=case, predicted_calls=predicted, elapsed_ms=elapsed_ms)
        result.selection_correct, result.arguments_correct = score_case(case, predicted)
        status = "PASS" if result.exact_match else "FAIL"
        print(f"    -> {render_calls(predicted)} [{status}]", flush=True)
        results.append(result)

    return build_report_lines(args, results)


def build_report_lines(args: argparse.Namespace, results: list[CaseResult]) -> list[str]:
    total = len(results)
    selection_hits = sum(1 for result in results if result.selection_correct)
    exact_hits = sum(1 for result in results if result.exact_match)
    tool_cases = [result for result in results if result.case.expected_calls]
    tool_selection_hits = [result for result in tool_cases if result.selection_correct]
    argument_hits = sum(1 for result in tool_selection_hits if result.arguments_correct)
    mean_latency = statistics.fmean(result.elapsed_ms for result in results)
    if tool_selection_hits:
        argument_row = (
            f"| Argument accuracy (of correct selections) | "
            f"{argument_hits}/{len(tool_selection_hits)} "
            f"({argument_hits / len(tool_selection_hits):.0%}) |"
        )
    else:
        argument_row = "| Argument accuracy (of correct selections) | n/a |"

    lines = [
        "# Tool Call Accuracy Eval",
        "",
        f"Runtime `{args.runtime}`, model `{args.model}`, device "
        f"`{args.device}`, max_tokens {args.max_tokens}. One completion per "
        "case; predicted calls parsed with "
        "`aibackends.core.tool_calls.extract_tool_calls`.",
        "",
        "## Environment",
        "",
        *environment_lines(("transformers", "torch", "llama-cpp-python")),
        "",
        "## Metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Tool selection accuracy | {selection_hits}/{total} "
        f"({selection_hits / total:.0%}) |",
        argument_row,
        f"| Exact match accuracy | {exact_hits}/{total} ({exact_hits / total:.0%}) |",
        f"| Mean latency per case | {mean_latency:,.0f} ms |",
        "",
        "## Cases",
        "",
        "| # | Question | Expected | Predicted | Selection | Arguments |",
        "|---|---|---|---|---|---|",
    ]
    for index, result in enumerate(results, start=1):
        selection = "pass" if result.selection_correct else "FAIL"
        arguments = "pass" if result.arguments_correct else "FAIL"
        lines.append(
            f"| {index} | {result.case.question} "
            f"| {render_calls(result.case.expected_calls)} "
            f"| {render_calls(result.predicted_calls)} "
            f"| {selection} | {arguments} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Tool selection counts 'no tool' cases: the model must answer",
            "  directly when no tool applies.",
            "- Arguments are compared after normalization (strings",
            "  case-insensitively, numbers numerically).",
            "- Latency includes the model's reasoning tokens, so it varies",
            "  with question complexity.",
        ]
    )
    return lines


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment_lines(extra_packages: tuple[str, ...] = ()) -> list[str]:
    lines = [
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Platform: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- aibackends: {_package_version('aibackends')}",
    ]
    for package in extra_packages:
        lines.append(f"- {package}: {_package_version(package)}")
    return lines


def write_report(name: str, lines: list[str]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"{date_prefix}_{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=["llamacpp", "transformers"], default="llamacpp")
    parser.add_argument("--model", default="lfm2.5-2.6b")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    try:
        lines = run_eval(args)
    except AIBackendsError as exc:
        raise SystemExit(f"Eval failed: {exc}") from exc

    report_path = write_report(f"tool-calls-{args.runtime}", lines)
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

"""Benchmark aibackends tasks per model, using the recommended model catalog.

For each catalog model: loads the model once (via ``preload()``, timed), then
runs each task ``--warm-calls`` times (default 10, configurable) against the
warm runtime. Chat and structured tasks (summarize, classify, extract) run
against ``--models``; the embed task runs against ``--embed-models``; the vl
task (image description, llama.cpp only) runs against ``--vl-models``. Pass
``all`` to any model list to expand it to every recommended catalog model in
that category for the runtime. Only catalog models are accepted, so committed
reports stay comparable across runs.

Writes a markdown report to ``benchmarks/reports/`` for committing to the repo.

With 10+ warm samples the report includes consistency statistics (stdev, CV,
p50/p95, drift) and per-segment trends that reveal inference degradation over
a sustained run; raise ``--warm-calls`` for a longer soak.

Usage:
    python benchmarks/benchmark_tasks.py --runtime transformers \
        --models gemma3-270m-it --warm-calls 10

    python benchmarks/benchmark_tasks.py --runtime llamacpp \
        --models all --embed-models all --vl-models all

Requires:
    pip install 'aibackends[transformers]'  # or [llamacpp]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Keep benchmark output readable; transformers is imported lazily by the runtime.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from _reporting import (
    TimingStats,
    benchmark_lock,
    consistency_table,
    ensure_gpu_dlls,
    environment_lines,
    segment_trend_lines,
    stats_table,
    write_report,
)
from pydantic import BaseModel

from aibackends import classify, embed, extract, summarize
from aibackends.core.config import clear_runtime_cache, get_runtime
from aibackends.core.exceptions import AIBackendsError
from aibackends.models import available_models, get_model_ref
from aibackends.runtimes import get_runtime_spec

SUMMARIZE_TEXT = (
    "AIBackends runs typed AI tasks on local runtimes. Models used to be "
    "reloaded on every task call, which dominated latency for short tasks. "
    "Runtime reuse keeps the loaded model in memory so subsequent calls only "
    "pay for inference, and preload() moves the load cost to startup."
)
CLASSIFY_TEXT = (
    "Invoice #4821 from Acme Corp: consulting services, subtotal $1,250.00, "
    "tax $0.00, total due $1,250.00 within 30 days."
)
CLASSIFY_LABELS = ["invoice", "contract", "receipt"]
EXTRACT_TEXT = "You can reach Alice Johnson at alice@example.com for onboarding."
EMBED_TEXT = "Local-first AI task execution with cached model runtimes."
VL_IMAGE = Path(__file__).parent.parent / "examples" / "data" / "images" / "receipt1.jpeg"
VL_PROMPT = "Describe this receipt in one sentence."


class ContactInfo(BaseModel):
    name: str
    email: str


TASK_RUNNERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "summarize": lambda overrides: summarize(SUMMARIZE_TEXT, **overrides),
    "classify": lambda overrides: classify(
        CLASSIFY_TEXT, labels=CLASSIFY_LABELS, **overrides
    ),
    "extract": lambda overrides: extract(EXTRACT_TEXT, schema=ContactInfo, **overrides),
}
CHAT_TASKS = ("summarize", "classify", "extract")
ALL_TASKS = (*CHAT_TASKS, "embed", "vl")
DEFAULT_CHAT_MODEL = {"transformers": "gemma3-270m-it", "llamacpp": "gemma4-e2b"}

# Recommended catalog models per category. Embedding and PII-only models are
# excluded from chat; VL needs the llama.cpp multimodal path (Gemma / Qwen VL).
EMBED_MODELS = ("bge-small", "minilm-l6")
NON_CHAT_MODELS = (*EMBED_MODELS, "openai-privacy")
VL_MODELS = ("gemma4-e2b", "gemma4-e4b", "qwen3-vl-4b", "qwen3-vl-8b")


def _resolve_catalog_model(name: str, runtime: str) -> str:
    catalog = available_models(runtime=runtime)
    if name not in catalog:
        supported = ", ".join(sorted(catalog)) or "none"
        raise SystemExit(
            f"Model '{name}' is not in the recommended catalog for runtime "
            f"'{runtime}'. Supported models: {supported}"
        )
    return name


def _expand_models(names: list[str], runtime: str, category: str) -> list[str]:
    if names != ["all"]:
        return [_resolve_catalog_model(name, runtime) for name in names]
    catalog = sorted(available_models(runtime=runtime))
    if category == "chat":
        return [name for name in catalog if name not in NON_CHAT_MODELS]
    if category == "embed":
        return [name for name in catalog if name in EMBED_MODELS]
    return [name for name in catalog if name in VL_MODELS]


def _task_overrides(runtime: str, model: str, max_tokens: int) -> dict[str, Any]:
    # Task functions take typed refs; get_runtime accepts plain strings.
    return {
        "runtime": get_runtime_spec(runtime),
        "model": get_model_ref(model),
        "max_tokens": max_tokens,
    }


def _benchmark_chat_model(
    *,
    runtime: str,
    model: str,
    tasks: list[str],
    warm_calls: int,
    max_tokens: int,
) -> list[str]:
    clear_runtime_cache()
    overrides = _task_overrides(runtime, model, max_tokens)

    load_stats = TimingStats("Model load (`preload()`)")
    started = time.perf_counter()
    try:
        get_runtime({"runtime": runtime, "model": model}).preload()
    except Exception as exc:  # keep an `all` sweep alive when one model fails
        print(f"  load failed: {exc}", flush=True)
        return [f"### `{model}`", "", f"Model load failed: {exc}", ""]
    load_stats.add((time.perf_counter() - started) * 1000)

    stats: list[TimingStats] = [load_stats]
    failures: list[str] = []
    for task_name in tasks:
        runner = TASK_RUNNERS[task_name]
        task_stats = TimingStats(f"`{task_name}` (warm)")
        print(f"  {task_name} x{warm_calls}...", flush=True)
        for _ in range(warm_calls):
            started = time.perf_counter()
            try:
                runner(overrides)
            except AIBackendsError as exc:
                failures.append(f"`{task_name}` failed: {exc}")
                break
            task_stats.add((time.perf_counter() - started) * 1000)
        if task_stats.samples_ms:
            stats.append(task_stats)

    lines = [f"### `{model}`", "", *stats_table(stats)]
    task_stats_only = stats[1:]
    consistency = consistency_table(task_stats_only)
    if consistency:
        lines.extend(["", "Consistency:", "", *consistency])
    for item in task_stats_only:
        trend = segment_trend_lines(item)
        if trend:
            lines.extend(["", *trend])
    if failures:
        lines.extend(["", "Failures:", *[f"- {failure}" for failure in failures]])
    lines.append("")
    return lines


def _benchmark_embed_model(
    *,
    runtime: str,
    model: str,
    warm_calls: int,
) -> list[str]:
    # preload() warms the generator, not the embedder, so for embeddings the
    # first call is timed separately as the load-including cold call.
    clear_runtime_cache()
    overrides = {"runtime": get_runtime_spec(runtime), "model": get_model_ref(model)}

    first_stats = TimingStats("First `embed` (includes model load)")
    warm_stats = TimingStats("`embed` (warm)")
    print(f"  embed 1 cold + {warm_calls} warm...", flush=True)
    try:
        started = time.perf_counter()
        embed(EMBED_TEXT, **overrides)
        first_stats.add((time.perf_counter() - started) * 1000)
        for _ in range(warm_calls):
            started = time.perf_counter()
            embed(EMBED_TEXT, **overrides)
            warm_stats.add((time.perf_counter() - started) * 1000)
    except AIBackendsError as exc:
        return [f"### `{model}` (embeddings)", "", f"Embed benchmark failed: {exc}", ""]

    lines = [f"### `{model}` (embeddings)", "", *stats_table([first_stats, warm_stats])]
    consistency = consistency_table([warm_stats])
    if consistency:
        lines.extend(["", "Consistency:", "", *consistency])
    trend = segment_trend_lines(warm_stats)
    if trend:
        lines.extend(["", *trend])
    lines.append("")
    return lines


def _benchmark_vl_model(
    *,
    runtime: str,
    model: str,
    warm_calls: int,
    max_tokens: int,
) -> list[str]:
    # Image inputs go through the runtime's multimodal client, which loads
    # separately from the text client, so the first call is timed as cold.
    clear_runtime_cache()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": VL_PROMPT},
                {"type": "image_url", "image_url": {"url": str(VL_IMAGE)}},
            ],
        }
    ]
    overrides: dict[str, Any] = {
        "runtime": runtime,
        "model": model,
        "max_tokens": max_tokens,
    }

    first_stats = TimingStats("First VL call (includes model load)")
    warm_stats = TimingStats("VL describe-image (warm)")
    print(f"  vl 1 cold + {warm_calls} warm...", flush=True)
    try:
        started = time.perf_counter()
        get_runtime(overrides).complete(messages)
        first_stats.add((time.perf_counter() - started) * 1000)
        for _ in range(warm_calls):
            started = time.perf_counter()
            get_runtime(overrides).complete(messages)
            warm_stats.add((time.perf_counter() - started) * 1000)
    except AIBackendsError as exc:
        return [f"### `{model}` (VL)", "", f"VL benchmark failed: {exc}", ""]

    lines = [
        f"### `{model}` (VL)",
        "",
        f"Prompt: {VL_PROMPT!r} with `{VL_IMAGE.name}`.",
        "",
        *stats_table([first_stats, warm_stats]),
    ]
    consistency = consistency_table([warm_stats])
    if consistency:
        lines.extend(["", "Consistency:", "", *consistency])
    trend = segment_trend_lines(warm_stats)
    if trend:
        lines.extend(["", *trend])
    lines.append("")
    return lines


def run_benchmark(args: argparse.Namespace) -> list[str]:
    chat_tasks = [name for name in args.tasks if name in CHAT_TASKS]
    if args.models is None:
        args.models = [DEFAULT_CHAT_MODEL.get(args.runtime, "gemma4-e2b")]
    chat_models = _expand_models(args.models, args.runtime, "chat") if chat_tasks else []
    run_embed = "embed" in args.tasks
    run_vl = "vl" in args.tasks

    lines = [
        "# Task Benchmark",
        "",
        f"Runtime `{args.runtime}`, {args.warm_calls} warm calls per task, "
        f"max_tokens {args.max_tokens}. Model load is paid once per model via "
        "`preload()`; task calls below run against the warm runtime.",
        "",
        "## Environment",
        "",
        *environment_lines(("transformers", "torch", "llama-cpp-python")),
        "",
        "## Results",
        "",
    ]

    if chat_tasks:
        for model in chat_models:
            print(f"Model {model} ({args.runtime}):", flush=True)
            lines.extend(
                _benchmark_chat_model(
                    runtime=args.runtime,
                    model=model,
                    tasks=chat_tasks,
                    warm_calls=args.warm_calls,
                    max_tokens=args.max_tokens,
                )
            )

    if run_embed:
        for embed_model in _expand_models(args.embed_models, args.runtime, "embed"):
            print(f"Model {embed_model} ({args.runtime}, embeddings):", flush=True)
            lines.extend(
                _benchmark_embed_model(
                    runtime=args.runtime,
                    model=embed_model,
                    warm_calls=args.warm_calls,
                )
            )

    if run_vl and args.runtime != "llamacpp":
        print("Skipping VL: image inputs require the llamacpp runtime.", flush=True)
        lines.extend(
            ["### VL", "", "Skipped: image inputs require the `llamacpp` runtime.", ""]
        )
    elif run_vl:
        for vl_model in _expand_models(args.vl_models, args.runtime, "vl"):
            print(f"Model {vl_model} ({args.runtime}, VL):", flush=True)
            lines.extend(
                _benchmark_vl_model(
                    runtime=args.runtime,
                    model=vl_model,
                    warm_calls=args.warm_calls,
                    max_tokens=args.max_tokens,
                )
            )

    lines.extend(
        [
            "## Notes",
            "",
            "- Structured tasks (`classify`, `extract`) include JSON parsing and",
            "  may retry on validation failures, so their timings can exceed a",
            "  single `complete()` call.",
            "- VL (image) inputs are supported by the llama.cpp runtime for",
            "  Gemma and Qwen VL GGUF models only.",
            "- Only models from the `aibackends.models` catalog are benchmarked.",
        ]
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="transformers")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Catalog models for chat/structured tasks, or 'all'. "
        "Defaults to the smallest chat model for the runtime.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(ALL_TASKS),
        choices=sorted(ALL_TASKS),
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--embed-models",
        nargs="+",
        default=["minilm-l6"],
        help="Catalog models for the embed task, or 'all'.",
    )
    parser.add_argument(
        "--vl-models",
        nargs="+",
        default=["qwen3-vl-4b"],
        help="Catalog models for the VL task, or 'all'. Requires llamacpp.",
    )
    parser.add_argument(
        "--warm-calls",
        type=int,
        default=10,
        help="Samples per task and model.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    ensure_gpu_dlls()
    try:
        with benchmark_lock():
            lines = run_benchmark(args)
    except AIBackendsError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    report_path = write_report(
        name=f"tasks-{args.runtime}",
        lines=lines,
        output_dir=args.output_dir,
    )
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

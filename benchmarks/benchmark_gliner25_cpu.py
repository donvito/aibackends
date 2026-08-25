"""Benchmark GLiNER 2.5 load cost and warm use-case latency on CPU.

Measures:

1. The first entity-extraction call, including model loading.
2. An explicit cold ``backend.load(device="cpu")``.
3. Warm entity extraction, constrained classification, joint IE, and
   long-document extraction.

Writes a markdown report to ``benchmarks/reports/`` for committing to the repo.

Usage:
    python benchmarks/benchmark_gliner25_cpu.py --warm-calls 10 --model gliner25-small

Requires:
    pip install 'aibackends[gliner25]'
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from _reporting import (
    TimingStats,
    benchmark_lock,
    consistency_table,
    environment_lines,
    segment_trend_lines,
    stats_table,
    write_report,
)

from aibackends.backends.extraction import get_extraction_backend
from aibackends.backends.extraction.gliner25 import clear_model_cache
from aibackends.core.exceptions import AIBackendsError

NER_TEXT = "Ada Lovelace wrote to Charles Babbage at Fastino Labs in London."
ROUTE_TEXT = "Delete the temporary cache files under /tmp/job-4821."
GRAPH_TEXT = (
    "Ada Lovelace leads Pioneer at Fastino Labs. Charles Babbage joined "
    "Fastino Labs last month. Fastino Labs is located in London."
)
LONG_TEXT = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "data"
    / "msa_contract.txt"
).read_text(encoding="utf-8")

ROUTE_TASKS = {
    "intent": {"labels": ["chat", "retrieve", "delete"]},
    "destination": {"labels": ["small_chat", "rag_tool", "file_tool"]},
}
ROUTE_CONSTRAINTS = [
    {"type": "implies", "if": ["intent", "delete"], "then": ["destination", "file_tool"]},
    {"type": "excludes", "left": ["intent", "chat"], "right": ["destination", "file_tool"]},
]
GRAPH_ENTITIES = ["person", "organization", "location"]
GRAPH_RELATIONS = [
    {
        "name": "works_for",
        "head": "person",
        "tail": "organization",
        "unique_head": True,
    },
    {"name": "located_in", "head": "organization", "tail": "location"},
]


def _time_call(call: Callable[[], Any]) -> float:
    started = time.perf_counter()
    call()
    return (time.perf_counter() - started) * 1000


def _clear_cache() -> None:
    clear_model_cache()
    gc.collect()


def _torch_cpu_threads() -> str:
    try:
        import torch
    except ImportError:
        return "unknown"
    return str(torch.get_num_threads())


def run_benchmark(args: argparse.Namespace) -> list[str]:
    backend = get_extraction_backend("gliner25")
    model = args.model
    first_ner = TimingStats("First entity extraction (load + inference)")
    cold_load = TimingStats("`backend.load(device=\"cpu\")`")
    warm_ner = TimingStats("Warm entity extraction")
    warm_classify = TimingStats("Warm constrained classification")
    warm_graph = TimingStats("Warm joint IE")
    warm_long = TimingStats("Warm long-document extraction")

    print(f"Benchmarking GLiNER 2.5 `{model}` on CPU", flush=True)

    print("1/5 first entity extraction including model load...", flush=True)
    _clear_cache()
    first_ner.add(
        _time_call(
            lambda: backend.extract_entities(
                NER_TEXT,
                ["person", "organization", "location"],
                device="cpu",
                model=model,
            )
        )
    )

    print("2/5 explicit cold load...", flush=True)
    _clear_cache()
    cold_load.add(_time_call(lambda: backend.load(device="cpu", model=model)))

    print(f"3/5 {args.warm_calls} warm NER / classify / graph calls...", flush=True)
    backend.extract_entities(
        NER_TEXT,
        ["person", "organization", "location"],
        device="cpu",
        model=model,
    )
    backend.classify_schema(
        ROUTE_TEXT,
        ROUTE_TASKS,
        device="cpu",
        model=model,
        constraints=ROUTE_CONSTRAINTS,
    )
    backend.extract_graph(
        GRAPH_TEXT,
        GRAPH_ENTITIES,
        GRAPH_RELATIONS,
        device="cpu",
        model=model,
    )
    for _ in range(args.warm_calls):
        warm_ner.add(
            _time_call(
                lambda: backend.extract_entities(
                    NER_TEXT,
                    ["person", "organization", "location"],
                    device="cpu",
                    model=model,
                )
            )
        )
        warm_classify.add(
            _time_call(
                lambda: backend.classify_schema(
                    ROUTE_TEXT,
                    ROUTE_TASKS,
                    device="cpu",
                    model=model,
                    constraints=ROUTE_CONSTRAINTS,
                )
            )
        )
        warm_graph.add(
            _time_call(
                lambda: backend.extract_graph(
                    GRAPH_TEXT,
                    GRAPH_ENTITIES,
                    GRAPH_RELATIONS,
                    device="cpu",
                    model=model,
                )
            )
        )

    print(f"4/5 {args.warm_calls} warm long-document extractions...", flush=True)
    backend.extract_entities(
        LONG_TEXT,
        ["person", "organization", "email"],
        device="cpu",
        model=model,
        long=True,
    )
    for _ in range(args.warm_calls):
        warm_long.add(
            _time_call(
                lambda: backend.extract_entities(
                    LONG_TEXT,
                    ["person", "organization", "email"],
                    device="cpu",
                    model=model,
                    long=True,
                )
            )
        )

    all_stats = [first_ner, cold_load, warm_ner, warm_classify, warm_graph, warm_long]
    speedup = first_ner.mean_ms / warm_ner.mean_ms if warm_ner.mean_ms else 0.0
    lines = [
        "# GLiNER 2.5 CPU Benchmark",
        "",
        f"Backend `{backend.name}`, model `{model}`, device forced to `cpu`, "
        f"{args.warm_calls} timed samples per warm scenario.",
        "",
        "## Environment",
        "",
        *environment_lines(("gliner2", "transformers", "torch", "protobuf")),
        f"- Logical CPUs: {os.cpu_count() or 'unknown'}",
        f"- Torch CPU threads: {_torch_cpu_threads()}",
        "",
        "## Results",
        "",
        *stats_table(all_stats),
        "",
        f"Warm entity extraction is **{speedup:,.1f}x** faster than the first "
        "call that includes model loading.",
        "",
        "## Consistency",
        "",
        *consistency_table([warm_ner, warm_classify, warm_graph, warm_long]),
        "",
        *segment_trend_lines(warm_ner),
        "",
        *segment_trend_lines(warm_classify),
        "",
        "## Notes",
        "",
        "- The benchmark forces `device=\"cpu\"`; no GPU inference path is used.",
        "- One untimed inference per scenario is run before collecting warm samples.",
        "- First-call timings include Python model construction from the local",
        "  Hugging Face cache; a first-ever network download is not measured.",
        "- Constrained classification and joint IE reuse the loaded extractor.",
        "- These timings measure performance, not extraction accuracy.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warm-calls",
        type=int,
        default=10,
        help="Timed samples per warm scenario.",
    )
    parser.add_argument(
        "--model",
        default="gliner25-small",
        help="Alias or Hub id to benchmark.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warm_calls < 1:
        parser.error("--warm-calls must be at least 1")

    try:
        with benchmark_lock():
            lines = run_benchmark(args)
    except AIBackendsError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    slug = args.model.replace("/", "-")
    report_path = write_report(
        name=f"{slug}-cpu",
        lines=lines,
        output_dir=args.output_dir,
    )
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

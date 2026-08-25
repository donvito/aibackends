"""Benchmark GLiNER2.5 model loading, extraction latency, and throughput on CPU.

Compares the small / base / multi checkpoints across:

1. Cold model load time.
2. Warm single-call latency per capability: entity extraction, entity
   extraction with span attributes, constrained classification, and joint
   entity-relation extraction.
3. Native batch throughput for entity extraction.
4. Long-document extraction latency versus document length (the blog's
   linear-scaling claim).

Writes a markdown report to ``benchmarks/reports/`` for committing to the repo.

Usage:
    python benchmarks/benchmark_gliner25.py --models small base multi --warm-calls 10

Requires:
    pip install 'aibackends[extraction]'
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

# Keep model download and Transformers logging readable.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from _reporting import (
    TimingStats,
    benchmark_lock,
    consistency_table,
    environment_lines,
    stats_table,
    write_report,
)

from aibackends.backends.extraction import get_extraction_backend
from aibackends.backends.extraction.gliner25 import clear_model_cache
from aibackends.core.exceptions import AIBackendsError

MODEL_PARAMS = {"small": "74M", "base": "194M", "multi": "287M"}

TEXTS = (
    "Apple hired Jane Doe as vice president of engineering in London last week.",
    "Please wire USD 4,500 to Northwind Analytics GmbH in Frankfurt by Friday.",
    "Dr. Elena Vasquez prescribed 400mg ibuprofen for the patient's headache.",
    "Tim Cook leads Apple in Cupertino while Sundar Pichai runs Google.",
    "The delivery from Beacon Retail Group arrives in Atlanta on March 3, 2026.",
    "Maria Gonzales can be reached at maria.gonzales@example.com or 404-555-0182.",
    "TotalEnergies signed an agreement with the government of Senegal in Dakar.",
    "Lufthansa opens a new hub in Munich, chief executive Carsten Spohr said.",
)
ENTITY_LABELS = ["person", "organization", "location", "date", "monetary amount"]
ATTRIBUTES = {
    "sentiment": {"labels": ["positive", "negative", "neutral"]},
}
CLASSIFY_TASKS = {
    "intent": {"labels": ["read", "write", "delete"]},
    "effects": {"labels": ["read_only", "create", "modify", "delete"], "multi_label": True},
}
CLASSIFY_CONSTRAINTS = [
    {"kind": "implies", "when": ["intent", "delete"], "then": ["effects", "delete"]},
    {"kind": "excludes", "when": ["intent", "read"], "then": ["effects", "delete"]},
]
GRAPH_ENTITIES = ["person", "organization", "location"]
GRAPH_RELATIONS = [
    {"name": "works_for", "head": "person", "tail": "organization", "unique_head": True},
    {"name": "located_in", "head": "organization", "tail": "location"},
]
LONG_DOCUMENT_WORDS = (250, 500, 1000, 2000)
LONG_DOCUMENT_SAMPLES = 3
CONTRACT_PATH = Path(__file__).parent.parent / "examples" / "data" / "sample_contract.txt"


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


def _long_document(words: int) -> str:
    base_words = CONTRACT_PATH.read_text(encoding="utf-8").split()
    repeated: list[str] = []
    while len(repeated) < words:
        repeated.extend(base_words)
    return " ".join(repeated[:words])


def _throughput_row(model: str, stats: TimingStats, batch_size: int) -> str:
    per_item_ms = stats.mean_ms / batch_size
    items_per_second = batch_size / (stats.mean_ms / 1000)
    return (
        f"| {model} | {batch_size} | {stats.mean_ms:,.1f} "
        f"| {per_item_ms:,.1f} | {items_per_second:,.1f} |"
    )


def benchmark_model(
    model: str, args: argparse.Namespace
) -> dict[str, TimingStats | dict[int, TimingStats]]:
    backend = get_extraction_backend("gliner2.5")
    device = args.device
    load = TimingStats(f"`{model}` cold load")
    entities = TimingStats(f"`{model}` warm entity extraction")
    attributes = TimingStats(f"`{model}` warm entities + span attributes")
    classify = TimingStats(f"`{model}` warm constrained classification")
    graph = TimingStats(f"`{model}` warm joint entity-relation graph")
    batch = TimingStats(f"`{model}` entity batch (size {args.batch_size})")
    long_docs: dict[int, TimingStats] = {
        words: TimingStats(f"`{model}` long document, {words} words")
        for words in LONG_DOCUMENT_WORDS
    }

    print(f"[{model}] 1/4 cold load...", flush=True)
    _clear_cache()
    load.add(_time_call(lambda: backend.load(model=model, device=device)))

    def run_entities(text: str) -> None:
        backend.extract_entities(text, labels=ENTITY_LABELS, model=model, device=device)

    def run_attributes(text: str) -> None:
        backend.extract_entities(
            text, labels=ENTITY_LABELS, attributes=ATTRIBUTES, model=model, device=device
        )

    def run_classify(text: str) -> None:
        backend.classify_text(
            text,
            tasks=CLASSIFY_TASKS,
            constraints=CLASSIFY_CONSTRAINTS,
            model=model,
            device=device,
        )

    def run_graph(text: str) -> None:
        backend.extract_graph(
            text,
            entities=GRAPH_ENTITIES,
            relations=GRAPH_RELATIONS,
            model=model,
            device=device,
        )

    print(f"[{model}] 2/4 {args.warm_calls} warm calls per capability...", flush=True)
    for warmup in (run_entities, run_attributes, run_classify, run_graph):
        warmup(TEXTS[0])
    for index in range(args.warm_calls):
        text = TEXTS[index % len(TEXTS)]
        entities.add(_time_call(lambda text=text: run_entities(text)))
        attributes.add(_time_call(lambda text=text: run_attributes(text)))
        classify.add(_time_call(lambda text=text: run_classify(text)))
        graph.add(_time_call(lambda text=text: run_graph(text)))

    print(
        f"[{model}] 3/4 {args.warm_calls} entity batches (size {args.batch_size})...",
        flush=True,
    )
    batch_texts = [TEXTS[index % len(TEXTS)] for index in range(args.batch_size)]
    backend.extract_entities_batch(
        batch_texts, labels=ENTITY_LABELS, model=model, device=device,
        batch_size=args.batch_size,
    )
    for _ in range(args.warm_calls):
        batch.add(
            _time_call(
                lambda: backend.extract_entities_batch(
                    batch_texts,
                    labels=ENTITY_LABELS,
                    model=model,
                    device=device,
                    batch_size=args.batch_size,
                )
            )
        )

    print(f"[{model}] 4/4 long documents {LONG_DOCUMENT_WORDS} words...", flush=True)
    for words, stats in long_docs.items():
        document = _long_document(words)
        backend.extract_entities(
            document, labels=ENTITY_LABELS, model=model, device=device, long_document=True
        )
        for _ in range(LONG_DOCUMENT_SAMPLES):
            stats.add(
                _time_call(
                    lambda document=document: backend.extract_entities(
                        document,
                        labels=ENTITY_LABELS,
                        model=model,
                        device=device,
                        long_document=True,
                    )
                )
            )

    return {
        "load": load,
        "entities": entities,
        "attributes": attributes,
        "classify": classify,
        "graph": graph,
        "batch": batch,
        "long_docs": long_docs,
    }


def run_benchmark(args: argparse.Namespace) -> list[str]:
    results: dict[str, dict[str, Any]] = {}
    for model in args.models:
        print(f"Benchmarking GLiNER2.5 `{model}` on {args.device}", flush=True)
        results[model] = benchmark_model(model, args)

    summary_rows = []
    for model, stats in results.items():
        summary_rows.append(
            f"| {model} | {MODEL_PARAMS.get(model, '?')} "
            f"| {stats['load'].mean_ms / 1000:,.1f} "
            f"| {stats['entities'].mean_ms:,.0f} "
            f"| {stats['attributes'].mean_ms:,.0f} "
            f"| {stats['classify'].mean_ms:,.0f} "
            f"| {stats['graph'].mean_ms:,.0f} "
            f"| {stats['batch'].mean_ms / args.batch_size:,.0f} |"
        )

    long_doc_rows = []
    for model, stats in results.items():
        cells = " | ".join(
            f"{stats['long_docs'][words].mean_ms:,.0f}" for words in LONG_DOCUMENT_WORDS
        )
        long_doc_rows.append(f"| {model} | {cells} |")

    all_warm_stats: list[TimingStats] = []
    for stats in results.values():
        all_warm_stats.extend(
            [stats["entities"], stats["attributes"], stats["classify"], stats["graph"]]
        )

    lines = [
        "# GLiNER2.5 CPU Benchmark",
        "",
        f"Backend `gliner2.5`, device forced to `{args.device}`, "
        f"{args.warm_calls} timed samples per warm scenario, "
        f"models: {', '.join(args.models)}.",
        "",
        "## Environment",
        "",
        *environment_lines(("gliner2", "transformers", "torch", "protobuf")),
        f"- Logical CPUs: {os.cpu_count() or 'unknown'}",
        f"- Torch CPU threads: {_torch_cpu_threads()}",
        "",
        "## Summary",
        "",
        "| Model | Params | Load (s) | Entities (ms) | +Attributes (ms) "
        "| Classify (ms) | Graph (ms) | Batch/item (ms) |",
        "|---|---|---|---|---|---|---|---|",
        *summary_rows,
        "",
        "Warm single-call means over short sentences; batch/item is the mean "
        "per-text latency inside one native batch.",
        "",
        "## Batch Throughput",
        "",
        "| Model | Batch size | Mean batch (ms) | Mean/item (ms) | Items/s |",
        "|---|---|---|---|---|",
        *[
            _throughput_row(model, stats["batch"], args.batch_size)
            for model, stats in results.items()
        ],
        "",
        "## Long-Document Scaling",
        "",
        f"Mean extraction latency (ms, {LONG_DOCUMENT_SAMPLES} samples) with "
        "`long_document=True`, chunk size 384 words, overlap 64:",
        "",
        "| Model | " + " | ".join(f"{words}w" for words in LONG_DOCUMENT_WORDS) + " |",
        "|---|" + "---|" * len(LONG_DOCUMENT_WORDS),
        *long_doc_rows,
        "",
        "## Detailed Results",
        "",
        *stats_table(
            [
                stats[key]
                for stats in results.values()
                for key in ("load", "entities", "attributes", "classify", "graph", "batch")
            ]
        ),
        "",
        "## Consistency",
        "",
        *consistency_table(all_warm_stats),
        "",
        "## Notes",
        "",
        f"- The benchmark forces `device={args.device!r}`; no GPU inference path is used.",
        "- One untimed inference per capability is run before collecting warm samples.",
        "- Cold-load timings construct the model from the local Hugging Face cache;",
        "  a first-ever network download is not measured.",
        "- Long documents are built by repeating a real contract to the target word",
        "  count, so chunk contents stay representative.",
        "- These timings measure performance, not extraction accuracy. Zero-shot",
        "  accuracy is covered by `evals/eval_gliner25.py`.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["small", "base", "multi"],
        help="Model variants to benchmark (small, base, multi, or HF repo ids).",
    )
    parser.add_argument(
        "--warm-calls",
        type=int,
        default=10,
        help="Timed samples per warm single and batch scenario.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of texts in each native batch.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, gpu, cuda, or mps.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warm_calls < 1:
        parser.error("--warm-calls must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    try:
        with benchmark_lock():
            lines = run_benchmark(args)
    except AIBackendsError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    report_path = write_report(
        name=f"gliner25-{args.device}",
        lines=lines,
        output_dir=args.output_dir,
    )
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

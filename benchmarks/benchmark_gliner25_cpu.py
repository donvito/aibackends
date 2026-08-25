"""Benchmark GLiNER2.5 use cases across the small, base, and multi checkpoints.

Measures local model construction plus warm entity, long-document, constrained
classification, Joint IE, combined-schema, and native batch inference on CPU.

Usage:
    python3 benchmarks/benchmark_gliner25_cpu.py \
        --models small base multi --warm-calls 10 --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _reporting import (  # noqa: E402
    TimingStats,
    benchmark_lock,
    consistency_table,
    environment_lines,
    stats_table,
    write_report,
)

from aibackends.backends.information_extraction import (  # noqa: E402
    BaseInformationExtractionBackend,
    get_information_extraction_backend,
)
from aibackends.backends.information_extraction.gliner25 import (  # noqa: E402
    GLINER25_MODEL_IDS,
    assert_source_spans,
    clear_gliner25_model_cache,
    normalize_device,
)
from aibackends.tasks import (  # noqa: E402
    batch_extract_entities,
    classify_schema,
    extract_entities,
    extract_entities_long,
    extract_graph,
    extract_schema,
)

CONTRACT_PATH = REPO_ROOT / "examples" / "data" / "contract.txt"
MODEL_IDS = GLINER25_MODEL_IDS
ENTITY_TEXT = "Apple CEO Tim Cook announced the iPhone 15 in Cupertino."
ENTITY_LABELS = ["company", "person", "product", "location"]
COMBINED_TEXT = (
    "Apple CEO Tim Cook announced the iPhone 15 for $999 in Cupertino. "
    "Reviewers praised the camera and called the launch exciting."
)
GRAPH_TEXT = (
    "Tim Cook leads Apple in Cupertino. "
    "Sundar Pichai runs Google in Mountain View."
)
BATCH_TEXTS = (
    "Apple CEO Tim Cook spoke in Cupertino.",
    "Microsoft CEO Satya Nadella presented Copilot in Seattle.",
    "Amazon CEO Andy Jassy announced new AWS services.",
    "Nvidia CEO Jensen Huang presented new chips in Taipei.",
    "Meta CEO Mark Zuckerberg discussed Llama in Menlo Park.",
    "OpenAI CEO Sam Altman spoke about ChatGPT in San Francisco.",
    "Tesla CEO Elon Musk announced a vehicle in Austin.",
    "IBM CEO Arvind Krishna discussed watsonx in New York.",
)


@dataclass
class ModelBenchmark:
    alias: str
    model_id: str
    load: TimingStats
    entity: TimingStats
    long_document: TimingStats
    classification: TimingStats
    joint_ie: TimingStats
    combined: TimingStats
    batch: TimingStats
    batch_size: int

    @property
    def warm_stats(self) -> list[TimingStats]:
        return [
            self.entity,
            self.long_document,
            self.classification,
            self.joint_ie,
            self.combined,
            self.batch,
        ]


def _time_call(call: Callable[[], object]) -> tuple[object, float]:
    started = time.perf_counter()
    result = call()
    return result, (time.perf_counter() - started) * 1000


def _measure(stats: TimingStats, call: Callable[[], object], samples: int) -> None:
    call()
    for _ in range(samples):
        _, elapsed_ms = _time_call(call)
        stats.add(elapsed_ms)


def _repeat_to_size(values: Sequence[str], size: int) -> list[str]:
    return [values[index % len(values)] for index in range(size)]


def _routing_schema(backend: BaseInformationExtractionBackend) -> Any:
    constraints = backend.classification_constraints
    return (
        backend.create_classification_schema()
        .single("task_type", ["summarization", "reasoning", "live_data"])
        .single("route", ["small_local_model", "large_reasoning_model", "tool_agent"])
        .constrain(
            constraints.implies(
                ("task_type", "summarization"),
                ("route", "small_local_model"),
            ),
            constraints.implies(
                ("task_type", "reasoning"),
                ("route", "large_reasoning_model"),
            ),
            constraints.implies(("task_type", "live_data"), ("route", "tool_agent")),
        )
    )


def _joint_schema(
    backend: BaseInformationExtractionBackend,
    *,
    model: str,
    device: str,
) -> Any:
    return (
        backend.create_joint_schema(model=model, device=device)
        .entities(["person", "organization", "location"])
        .relation("works_for", "person", "organization", unique_head=True)
        .relation("located_in", "organization", "location", unique_head=True)
        .no_self_loops()
    )


def _combined_schema(
    backend: BaseInformationExtractionBackend,
    *,
    model: str,
    device: str,
) -> Any:
    return (
        backend.create_schema(model=model, device=device)
        .entities(["person", "company", "product", "location"])
        .classification("sentiment", ["positive", "negative", "neutral"])
        .classification("document_type", ["product_news", "review", "opinion"])
        .relations(["works_for", "announced_by", "located_in"])
        .structure("product")
        .field("name", dtype="str")
        .field("price", dtype="str")
        .field("feature", dtype="list")
    )


def run_model(
    alias: str,
    *,
    device: str,
    warm_calls: int,
    batch_size: int,
) -> ModelBenchmark:
    clear_gliner25_model_cache()
    gc.collect()
    backend = get_information_extraction_backend("gliner25")
    model_id = MODEL_IDS[alias]
    print(f"\nLoading {alias}: {model_id}", flush=True)
    load = TimingStats(f"{alias}: model load")
    started = time.perf_counter()
    backend.load(model=alias, device=device)
    load.add((time.perf_counter() - started) * 1000)

    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    classification_schema = _routing_schema(backend)
    classification_config = backend.create_classification_config(
        decoder="auto",
        on_infeasible="raise",
    )
    joint_schema = _joint_schema(backend, model=alias, device=device)
    joint_config = backend.create_joint_config(optimizer="beam", beam_size=32)
    combined_schema = _combined_schema(backend, model=alias, device=device)
    batch_texts = _repeat_to_size(BATCH_TEXTS, batch_size)

    def entity_call() -> object:
        return extract_entities(
            ENTITY_TEXT,
            ENTITY_LABELS,
            backend=backend.name,
            model=alias,
            device=device,
            include_spans=True,
            include_confidence=True,
        )

    def long_document_call() -> object:
        return extract_entities_long(
            contract,
            {
                "person": "Names of contract parties",
                "email": "Email addresses",
                "phone_number": "Telephone numbers",
                "obligation": "Complete clauses describing a required action",
                "termination_clause": "Complete clauses describing contract termination",
            },
            backend=backend.name,
            model=alias,
            device=device,
            chunk_size=128,
            chunk_overlap=32,
            include_spans=True,
            include_confidence=True,
        )

    def classification_call() -> object:
        return classify_schema(
            "Look up the current weather in Singapore.",
            classification_schema,
            backend=backend.name,
            model=alias,
            device=device,
            config=classification_config,
        )

    def graph_call() -> object:
        return extract_graph(
            GRAPH_TEXT,
            joint_schema,
            backend=backend.name,
            model=alias,
            device=device,
            config=joint_config,
        )

    def combined_call() -> object:
        return extract_schema(
            COMBINED_TEXT,
            combined_schema,
            backend=backend.name,
            model=alias,
            device=device,
            include_spans=True,
            include_confidence=True,
        )

    def batch_call() -> object:
        return batch_extract_entities(
            batch_texts,
            ENTITY_LABELS,
            backend=backend.name,
            model=alias,
            device=device,
            batch_size=batch_size,
            include_spans=True,
            include_confidence=True,
        )

    print(f"  validating {alias} outputs...", flush=True)
    assert_source_spans(ENTITY_TEXT, entity_call())
    assert_source_spans(contract, long_document_call())
    assert_source_spans(GRAPH_TEXT, graph_call())
    assert_source_spans(COMBINED_TEXT, combined_call())
    for text, result in zip(batch_texts, batch_call(), strict=True):
        assert_source_spans(text, result)

    entity = TimingStats(f"{alias}: entity extraction")
    long_document = TimingStats(f"{alias}: long-document extraction")
    classification = TimingStats(f"{alias}: constrained classification")
    joint_ie = TimingStats(f"{alias}: Joint IE")
    combined = TimingStats(f"{alias}: combined schema")
    batch = TimingStats(f"{alias}: entity batch (size {batch_size})")

    scenarios = (
        ("entity extraction", entity, entity_call),
        ("long-document extraction", long_document, long_document_call),
        ("constrained classification", classification, classification_call),
        ("Joint IE", joint_ie, graph_call),
        ("combined schema", combined, combined_call),
        (f"entity batch size {batch_size}", batch, batch_call),
    )
    for name, stats, call in scenarios:
        print(f"  timing {name}: {warm_calls} calls", flush=True)
        _measure(stats, call, warm_calls)

    result = ModelBenchmark(
        alias=alias,
        model_id=model_id,
        load=load,
        entity=entity,
        long_document=long_document,
        classification=classification,
        joint_ie=joint_ie,
        combined=combined,
        batch=batch,
        batch_size=batch_size,
    )
    clear_gliner25_model_cache()
    gc.collect()
    return result


def _throughput_row(result: ModelBenchmark) -> str:
    per_item_ms = result.batch.mean_ms / result.batch_size
    items_per_second = result.batch_size / (result.batch.mean_ms / 1000)
    return (
        f"| `{result.alias}` | {result.batch_size} | {result.batch.mean_ms:,.1f} | "
        f"{per_item_ms:,.1f} | {items_per_second:,.1f} |"
    )


def build_report(
    results: list[ModelBenchmark],
    *,
    device: str,
    warm_calls: int,
) -> list[str]:
    all_stats = [stats for result in results for stats in [result.load, *result.warm_stats]]
    warm_stats = [stats for result in results for stats in result.warm_stats]
    lines = [
        "# GLiNER2.5 CPU Use-Case Benchmark",
        "",
        f"Models run sequentially through the aibackends `gliner25` backend on `{device}` "
        f"with {warm_calls} timed warm calls per scenario.",
        "",
        "## Environment",
        "",
        *environment_lines(("gliner2", "transformers", "torch", "protobuf")),
        f"- Logical CPUs: {os.cpu_count() or 'unknown'}",
        "",
        "## Latency",
        "",
        *stats_table(all_stats),
        "",
        "## Native Batch Throughput",
        "",
        "| Model | Batch size | Mean batch (ms) | Mean/item (ms) | Items/s |",
        "|---|---:|---:|---:|---:|",
        *[_throughput_row(result) for result in results],
        "",
        "## Consistency",
        "",
        *consistency_table(warm_stats),
        "",
        "## Notes",
        "",
        "- Every model is loaded and benchmarked sequentially to limit resident memory.",
        "- Load timings use files already present in the local Hugging Face cache.",
        "- Each warm scenario runs one untimed call before collecting samples.",
        "- Long-document extraction scans the sample contract in overlapping 128-word chunks.",
        "- Joint IE uses beam decoding with a beam size of 32.",
        "- Timings include schema decoding and result formatting but exclude result validation.",
        "- These numbers measure performance; use the applied eval report for fixture quality.",
    ]
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_IDS),
        default=list(MODEL_IDS),
        help="Model aliases to benchmark sequentially.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warm-calls", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warm_calls < 1:
        parser.error("--warm-calls must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    device = normalize_device(args.device)
    with benchmark_lock():
        results = [
            run_model(
                alias,
                device=device,
                warm_calls=args.warm_calls,
                batch_size=args.batch_size,
            )
            for alias in args.models
        ]
    report_path = write_report(
        name=f"gliner25-{device}",
        lines=build_report(results, device=device, warm_calls=args.warm_calls),
        output_dir=args.output_dir,
    )
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

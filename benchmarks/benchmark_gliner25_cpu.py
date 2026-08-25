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

from examples.gliner25.common import (  # noqa: E402
    MODEL_IDS,
    assert_source_spans,
    normalize_device,
)

CONTRACT_PATH = REPO_ROOT / "examples" / "data" / "contract.txt"
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


def _routing_schema() -> Any:
    from gliner2.classification import ClassificationSchema
    from gliner2.classification import constraints as C

    return (
        ClassificationSchema()
        .single("task_type", ["summarization", "reasoning", "live_data"])
        .single("route", ["small_local_model", "large_reasoning_model", "tool_agent"])
        .constrain(
            C.implies(("task_type", "summarization"), ("route", "small_local_model")),
            C.implies(("task_type", "reasoning"), ("route", "large_reasoning_model")),
            C.implies(("task_type", "live_data"), ("route", "tool_agent")),
        )
    )


def _joint_schema(joint: Any) -> Any:
    return (
        joint.create_schema()
        .entities(["person", "organization", "location"])
        .relation("works_for", "person", "organization", unique_head=True)
        .relation("located_in", "organization", "location", unique_head=True)
        .no_self_loops()
    )


def _combined_schema(model: Any) -> Any:
    return (
        model.create_schema()
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
    from gliner2 import AutoExtractor
    from gliner2.classification import ClassificationConfig, Classifier
    from gliner2.joint_ie import JointIE, JointIEConfig

    gc.collect()
    model_id = MODEL_IDS[alias]
    print(f"\nLoading {alias}: {model_id}", flush=True)
    load = TimingStats(f"{alias}: model load")
    started = time.perf_counter()
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    model.eval()
    load.add((time.perf_counter() - started) * 1000)

    contract = CONTRACT_PATH.read_text(encoding="utf-8")
    classifier = Classifier(model, device=device).eval()
    classification_schema = _routing_schema()
    classification_config = ClassificationConfig(decoder="auto", on_infeasible="raise")
    joint = JointIE(model, device=device).eval()
    joint_schema = _joint_schema(joint)
    joint_config = JointIEConfig(optimizer="beam", beam_size=32)
    combined_schema = _combined_schema(model)
    batch_texts = _repeat_to_size(BATCH_TEXTS, batch_size)

    def extract_entities() -> object:
        return model.extract_entities(
            ENTITY_TEXT,
            ENTITY_LABELS,
            include_spans=True,
            include_confidence=True,
        )

    def extract_long_document() -> object:
        return model.extract_entities_long(
            contract,
            {
                "person": "Names of contract parties",
                "email": "Email addresses",
                "phone_number": "Telephone numbers",
                "obligation": "Complete clauses describing a required action",
                "termination_clause": "Complete clauses describing contract termination",
            },
            chunk_size=128,
            chunk_overlap=32,
            include_spans=True,
            include_confidence=True,
        )

    def classify_route() -> object:
        return classifier.classify(
            "Look up the current weather in Singapore.",
            classification_schema,
            config=classification_config,
        )

    def extract_graph() -> object:
        return joint.extract(GRAPH_TEXT, joint_schema, config=joint_config)

    def extract_combined() -> object:
        return model.extract(
            COMBINED_TEXT,
            combined_schema,
            include_spans=True,
            include_confidence=True,
        )

    def extract_batch() -> object:
        return model.batch_extract_entities(
            batch_texts,
            ENTITY_LABELS,
            batch_size=batch_size,
            include_spans=True,
            include_confidence=True,
        )

    print(f"  validating {alias} outputs...", flush=True)
    assert_source_spans(ENTITY_TEXT, extract_entities())
    assert_source_spans(contract, extract_long_document())
    assert_source_spans(GRAPH_TEXT, extract_graph())
    assert_source_spans(COMBINED_TEXT, extract_combined())
    for text, result in zip(batch_texts, extract_batch(), strict=True):
        assert_source_spans(text, result)

    entity = TimingStats(f"{alias}: entity extraction")
    long_document = TimingStats(f"{alias}: long-document extraction")
    classification = TimingStats(f"{alias}: constrained classification")
    joint_ie = TimingStats(f"{alias}: Joint IE")
    combined = TimingStats(f"{alias}: combined schema")
    batch = TimingStats(f"{alias}: entity batch (size {batch_size})")

    scenarios = (
        ("entity extraction", entity, extract_entities),
        ("long-document extraction", long_document, extract_long_document),
        ("constrained classification", classification, classify_route),
        ("Joint IE", joint_ie, extract_graph),
        ("combined schema", combined, extract_combined),
        (f"entity batch size {batch_size}", batch, extract_batch),
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
        f"Models run sequentially on `{device}` with {warm_calls} timed warm calls per scenario.",
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

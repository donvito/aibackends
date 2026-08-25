"""Compare GLiNER 2.5 model variants across representative CPU use cases.

This is a local latency and behavior-smoke benchmark, not a reproduction of
Fastino's 16-dataset macro-F1 evaluation.

Usage:
    python benchmarks/benchmark_gliner25_cpu.py --warm-calls 3
    python benchmarks/benchmark_gliner25_cpu.py --models small base

Requires:
    pip install 'aibackends[information-extraction]'
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep model download and Transformers logging readable.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings(
    "ignore",
    message="Checkpoint uses legacy list-valued extra_special_tokens metadata",
)
warnings.filterwarnings(
    "ignore",
    message="Encoder rejected attn_implementation='sdpa'",
)

from _reporting import TimingStats, benchmark_lock, environment_lines, write_report
from gliner2 import AttributeGroup, AutoExtractor
from gliner2.classification import Classifier, ClassificationSchema
from gliner2.classification import constraints as C
from gliner2.joint_ie import JointIE, JointIEConfig


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    model_id: str
    parameters_m: float
    intended_use: str


@dataclass
class ModelResult:
    spec: ModelSpec
    load_ms: float
    timings: dict[str, TimingStats]
    checks: dict[str, tuple[bool, str]]


MODEL_SPECS = {
    "small": ModelSpec(
        "small",
        "fastino/gliner2.5-small-v1",
        73.9,
        "Fast English CPU / edge",
    ),
    "base": ModelSpec(
        "base",
        "fastino/gliner2.5-base-v1",
        193.6,
        "Default English multi-task",
    ),
    "multi": ModelSpec(
        "multi",
        "fastino/gliner2.5-multi-v1",
        287.4,
        "Default multilingual multi-task",
    ),
}

ENTITY_TEXT = (
    "Northstar Analytics LLC signed an agreement with Blue Harbor Bank on August 18, 2026."
)
INVOICE_TEXT = (
    "Invoice INV-2048 from Acme Labs totals $1,240.00 and is due September 30, 2026."
)
FEEDBACK_TEXT = "The Atlas camera is excellent, but its battery life is disappointing."
GRAPH_TEXT = "Maya Chen leads Northstar Analytics in Singapore."
BATCH_TEXTS = (
    "Apple CEO Tim Cook announced Vision Pro updates in Cupertino.",
    "Maya Chen joined Northstar Analytics in Singapore.",
    "Acme Labs opened an office in Toronto.",
    "Blue Harbor Bank appointed Jordan Lee as chief risk officer.",
)
LONG_PADDING = " ".join(
    [
        "The report contains background, definitions, service levels, and reporting terms."
    ]
    * 6
)
LONG_TEXT = (
    f"{LONG_PADDING} Northstar Analytics LLC signed this services agreement with "
    "Blue Harbor Bank on August 18, 2026. Either party may terminate this Agreement for "
    "convenience by giving the other party at least thirty (30) days' prior written notice. "
    f"{LONG_PADDING}"
)

ENTITY_SCHEMA = {
    "contract party": "Organizations that are parties to the agreement",
    "effective date": "The date the agreement starts",
}
INVOICE_SCHEMA = {
    "invoice": [
        "invoice_number::str",
        "vendor::str",
        "total::str",
        "due_date::str",
    ]
}


def _time_call(call: Callable[[], Any]) -> tuple[float, Any]:
    started = time.perf_counter()
    result = call()
    return (time.perf_counter() - started) * 1000, result


def _entity_groups(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups = result.get("entities", {})
    if not isinstance(groups, dict):
        return {}
    return {
        label: items
        for label, items in groups.items()
        if isinstance(label, str) and isinstance(items, list)
    }


def _offsets_are_valid(text: str, result: dict[str, Any]) -> bool:
    for entities in _entity_groups(result).values():
        for entity in entities:
            if "start" not in entity or "end" not in entity:
                continue
            if text[entity["start"] : entity["end"]] != entity.get("text"):
                return False
    return True


def _check_entities(result: Any) -> tuple[bool, str]:
    groups = _entity_groups(result)
    parties = {item.get("text") for item in groups.get("contract party", [])}
    dates = {item.get("text") for item in groups.get("effective date", [])}
    expected_parties = {"Northstar Analytics LLC", "Blue Harbor Bank"}
    passed = expected_parties <= parties and "August 18, 2026" in dates
    passed = passed and _offsets_are_valid(ENTITY_TEXT, result)
    return passed, f"parties={sorted(parties)!r}; dates={sorted(dates)!r}"


def _check_invoice(result: Any) -> tuple[bool, str]:
    invoices = result.get("invoice", [])
    record = invoices[0] if invoices else {}
    expected = {
        "invoice_number": "INV-2048",
        "vendor": "Acme Labs",
        "total": "$1,240.00",
        "due_date": "September 30, 2026",
    }
    return record == expected, f"record={record!r}"


def _check_attributes(result: Any) -> tuple[bool, str]:
    products = _entity_groups(result).get("product", [])
    atlas = next((item for item in products if item.get("text") == "Atlas camera"), None)
    sentiment = atlas.get("sentiment", {}) if atlas else {}
    passed = bool(atlas) and sentiment.get("label") == "positive"
    passed = passed and _offsets_are_valid(FEEDBACK_TEXT, result)
    return passed, f"Atlas camera sentiment={sentiment.get('label')!r}"


def _check_routing(result: Any) -> tuple[bool, str]:
    route = result.value("route")
    effects = result.selected("effects")
    passed = result.feasible and route == "delete" and "delete" in effects
    return passed, f"route={route!r}; effects={effects!r}; feasible={result.feasible}"


def _check_graph(result: Any) -> tuple[bool, str]:
    entities = {entity.id: entity for entity in result.entities}
    expected_types = {
        "works_for": ("person", "organization"),
        "located_in": ("organization", "location"),
    }
    relation_types = {relation.type for relation in result.relations}
    typed = all(
        relation.head in entities
        and relation.tail in entities
        and relation.head != relation.tail
        and (
            entities[relation.head].type,
            entities[relation.tail].type,
        )
        == expected_types[relation.type]
        for relation in result.relations
        if relation.type in expected_types
    )
    passed = result.feasible and set(expected_types) <= relation_types and typed
    return passed, f"relations={sorted(relation_types)!r}; feasible={result.feasible}"


def _check_batch(result: Any) -> tuple[bool, str]:
    if len(result) != len(BATCH_TEXTS):
        return False, f"returned {len(result)} results for {len(BATCH_TEXTS)} texts"
    offset_checks = [
        _offsets_are_valid(text, item) for text, item in zip(BATCH_TEXTS, result, strict=True)
    ]
    entity_counts = [
        sum(len(items) for items in _entity_groups(item).values()) for item in result
    ]
    return all(offset_checks) and all(entity_counts), f"entities per text={entity_counts!r}"


def _check_long_document(result: Any) -> tuple[bool, str]:
    groups = _entity_groups(result)
    parties = {item.get("text") for item in groups.get("contract party", [])}
    dates = {item.get("text") for item in groups.get("effective date", [])}
    passed = {"Northstar Analytics LLC", "Blue Harbor Bank"} <= parties
    passed = passed and "August 18, 2026" in dates
    passed = passed and _offsets_are_valid(LONG_TEXT, result)
    count = sum(len(items) for items in groups.values())
    return passed, f"{count} spans with valid global offsets={_offsets_are_valid(LONG_TEXT, result)}"


def _build_scenarios(
    model: Any,
) -> tuple[
    dict[str, Callable[[], Any]],
    dict[str, Callable[[Any], tuple[bool, str]]],
]:
    attribute_schema = (
        model.create_schema()
        .entities(["product"])
        .entity_attributes(
            {
                "sentiment": AttributeGroup(
                    ["positive", "negative", "neutral"],
                    applies_to=["product"],
                    qualify_labels=True,
                )
            }
        )
    )
    classifier = Classifier(model)
    routing_schema = (
        ClassificationSchema()
        .single("route", ["read", "write", "delete"])
        .multi(
            "effects",
            ["read_only", "create", "modify", "delete"],
            min_labels=1,
            max_labels=2,
        )
        .constrain(
            C.implies(("route", "delete"), ("effects", "delete")),
            C.implies(("route", "read"), ("effects", "read_only")),
            C.excludes(("route", "read"), ("effects", "delete")),
            C.excludes(("route", "read"), ("effects", "modify")),
        )
    )
    joint = JointIE(model)
    graph_schema = (
        joint.create_schema()
        .entities(["person", "organization", "location"])
        .relation("works_for", "person", "organization", unique_head=True)
        .relation("located_in", "organization", "location", unique_head=True)
        .no_self_loops()
    )

    scenarios = {
        "Entities": lambda: model.extract_entities(
            ENTITY_TEXT,
            ENTITY_SCHEMA,
            include_spans=True,
            include_confidence=True,
        ),
        "Structured JSON": lambda: model.extract_json(INVOICE_TEXT, INVOICE_SCHEMA),
        "Span attributes": lambda: model.extract(
            FEEDBACK_TEXT,
            attribute_schema,
            include_spans=True,
            include_confidence=True,
        ),
        "Constrained routing": lambda: classifier.classify(
            "Delete the temporary file from /tmp.",
            routing_schema,
        ),
        "Joint IE graph": lambda: joint.extract(
            GRAPH_TEXT,
            graph_schema,
            config=JointIEConfig(optimizer="beam", beam_size=16),
        ),
        "Batch entities (4)": lambda: model.batch_extract_entities(
            list(BATCH_TEXTS),
            ["person", "organization", "product", "location"],
            include_spans=True,
            include_confidence=True,
            batch_size=4,
        ),
        "Long document": lambda: model.extract_entities_long(
            LONG_TEXT,
            ENTITY_SCHEMA,
            chunk_size=64,
            chunk_overlap=20,
            include_spans=True,
            include_confidence=True,
        ),
    }
    checks = {
        "Entities": _check_entities,
        "Structured JSON": _check_invoice,
        "Span attributes": _check_attributes,
        "Constrained routing": _check_routing,
        "Joint IE graph": _check_graph,
        "Batch entities (4)": _check_batch,
        "Long document": _check_long_document,
    }
    return scenarios, checks


def _run_model(spec: ModelSpec, warm_calls: int) -> ModelResult:
    print(f"Loading {spec.model_id}...", flush=True)
    load_ms, model = _time_call(
        lambda: AutoExtractor.from_pretrained(spec.model_id, map_location="cpu")
    )
    scenarios, validators = _build_scenarios(model)
    timings: dict[str, TimingStats] = {}
    checks: dict[str, tuple[bool, str]] = {}

    for index, (name, call) in enumerate(scenarios.items(), start=1):
        print(f"  {index}/{len(scenarios)} {name}: warm-up and {warm_calls} samples", flush=True)
        _, warm_result = _time_call(call)
        checks[name] = validators[name](warm_result)
        stats = TimingStats(name)
        for _ in range(warm_calls):
            elapsed_ms, _ = _time_call(call)
            stats.add(elapsed_ms)
        timings[name] = stats

    return ModelResult(spec=spec, load_ms=load_ms, timings=timings, checks=checks)


def _latency_table(results: list[ModelResult]) -> list[str]:
    lines = [
        "| Model | Scenario | Samples | Mean (ms) | p50 (ms) | Min (ms) | Max (ms) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for stats in result.timings.values():
            lines.append(
                f"| {result.spec.alias} | {stats.label} | {len(stats.samples_ms)} "
                f"| {stats.mean_ms:,.1f} | {stats.percentile_ms(0.5):,.1f} "
                f"| {stats.min_ms:,.1f} | {stats.max_ms:,.1f} |"
            )
    return lines


def _validation_table(results: list[ModelResult]) -> list[str]:
    lines = [
        "| Model | Scenario | Result | Observation |",
        "|---|---|---|---|",
    ]
    for result in results:
        for scenario, (passed, detail) in result.checks.items():
            escaped_detail = detail.replace("|", "\\|")
            lines.append(
                f"| {result.spec.alias} | {scenario} | {'PASS' if passed else 'MISS'} "
                f"| {escaped_detail} |"
            )
    return lines


def _summary_table(results: list[ModelResult]) -> list[str]:
    lines = [
        "| Model | Parameters | Load (s) | Mean warm scenario (ms) "
        "| Batch items/s | Checks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        average_ms = statistics.fmean(stats.mean_ms for stats in result.timings.values())
        batch_ms = result.timings["Batch entities (4)"].mean_ms
        batch_items_per_second = 4 / (batch_ms / 1000)
        passed = sum(check[0] for check in result.checks.values())
        lines.append(
            f"| {result.spec.alias} | {result.spec.parameters_m:.1f}M "
            f"| {result.load_ms / 1000:,.2f} | {average_ms:,.1f} "
            f"| {batch_items_per_second:,.2f} | {passed}/{len(result.checks)} |"
        )
    return lines


def _report_lines(results: list[ModelResult], warm_calls: int) -> list[str]:
    fastest = min(
        results,
        key=lambda result: statistics.fmean(
            stats.mean_ms for stats in result.timings.values()
        ),
    )
    return [
        "# GLiNER 2.5 CPU Benchmark",
        "",
        "Local model-construction latency, warm inference latency, batch throughput, and",
        "small behavior checks across the three public GLiNER 2.5 boundary checkpoints.",
        "This is **not** a reproduction of the release's 16-dataset macro-F1 evaluation.",
        "",
        "## Environment",
        "",
        *environment_lines(("gliner2", "transformers", "torch", "huggingface-hub")),
        f"- Logical CPUs: {os.cpu_count() or 'unknown'}",
        f"- Timed samples per scenario: {warm_calls}",
        "",
        "## Local Summary",
        "",
        *_summary_table(results),
        "",
        f"`{fastest.spec.alias}` has the lowest unweighted mean latency across this local",
        "scenario mix. The checks only validate expected behavior on seven fixed examples;",
        "they do not establish general model accuracy.",
        "",
        "## Warm Latency",
        "",
        *_latency_table(results),
        "",
        "Batch latency is the total time for four texts. The summary converts that mean to",
        "items/second; other rows process one input per call.",
        "",
        "## Behavior Checks",
        "",
        "These checks cover expected entities/records, span-to-source offset integrity,",
        "constraint feasibility, and typed graph edges. `MISS` is reported rather than",
        "failing the benchmark because zero-shot model quality is itself an observed result.",
        "",
        *_validation_table(results),
        "",
        "## Published Accuracy Context",
        "",
        "Fastino reports the following macro-F1 task averages in the",
        "[GLiNER 2.5 release post](https://fastino.ai/blog/"
        "gliner2-5-span-free-information-extraction). These are published numbers, not",
        "measurements from this script.",
        "",
        "| Model | Overall | Classification | Extraction |",
        "|---|---:|---:|---:|",
        "| multi | 56.17 | 72.44 | 46.40 |",
        "| base | 54.87 | 69.86 | 45.88 |",
        "| small | Not reported | Not reported | Not reported |",
        "",
        "## Model Selection",
        "",
        "- `small` (73.9M): start here for CPU, edge, and latency-sensitive English work.",
        "- `base` (193.6M): default for English when quality matters more than minimum latency.",
        "- `multi` (287.4M): use for multilingual inputs; it also has the strongest published",
        "  overall average among the GLiNER 2.5 variants in the release post.",
        "",
        "## Notes",
        "",
        "- Inference is forced to CPU and models are run sequentially under the shared",
        "  benchmark lock.",
        "- Each scenario gets one untimed warm-up before timing.",
        "- Model-construction time excludes network transfer only when weights are already in",
        "  the Hugging Face cache. Pre-download models before comparing load times.",
        "- Long-document extraction uses overlapping 64-word chunks with global source",
        "  offsets; it is intentionally more expensive than a one-window call.",
        "- Joint IE uses beam search with typed endpoints, unique heads, and no self-loops.",
    ]


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
        help="Model aliases to benchmark in order.",
    )
    parser.add_argument(
        "--warm-calls",
        type=int,
        default=3,
        help="Timed samples per model and scenario.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.warm_calls < 1:
        parser.error("--warm-calls must be at least 1")

    with benchmark_lock():
        results = []
        for alias in args.models:
            results.append(_run_model(MODEL_SPECS[alias], args.warm_calls))
            _release_model_memory()

    report_path = write_report(
        name="gliner25-cpu",
        lines=_report_lines(results, args.warm_calls),
        output_dir=args.output_dir,
    )
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

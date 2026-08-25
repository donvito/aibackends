"""Evaluate practical GLiNER2.5 use cases on a small labeled fixture set.

This is a repository-sized applied evaluation, not a reproduction of Fastino's
16-dataset research benchmark. It measures exact source-grounded entities,
constrained classification, typed relation graphs, and span attributes.

Usage:
    python3 evals/eval_gliner25.py --models small base multi --device cpu
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks._reporting import environment_lines, slugify, write_report  # noqa: E402
from examples.gliner25.common import (  # noqa: E402
    MODEL_IDS,
    assert_source_spans,
    normalize_device,
    result_to_dict,
)

FIXTURE_PATH = Path(__file__).parent / "data" / "gliner25_cases.json"
REPORTS_DIR = Path(__file__).parent / "reports"


@dataclass
class PRFCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add(self, expected: set[tuple[str, ...]], predicted: set[tuple[str, ...]]) -> None:
        self.true_positive += len(expected & predicted)
        self.false_positive += len(predicted - expected)
        self.false_negative += len(expected - predicted)

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass(frozen=True)
class CaseRecord:
    category: str
    case_id: str
    expected: object
    predicted: object
    exact: bool
    offsets_valid: bool | None = None


@dataclass
class ModelEval:
    alias: str
    model_id: str
    load_seconds: float
    entity: PRFCounts = field(default_factory=PRFCounts)
    relation: PRFCounts = field(default_factory=PRFCounts)
    classification_hits: int = 0
    classification_total: int = 0
    attribute_hits: int = 0
    attribute_total: int = 0
    feasible_hits: int = 0
    feasible_total: int = 0
    graph_valid_hits: int = 0
    graph_valid_total: int = 0
    offset_hits: int = 0
    offset_total: int = 0
    records: list[CaseRecord] = field(default_factory=list)

    @property
    def classification_accuracy(self) -> float:
        if not self.classification_total:
            return 0.0
        return self.classification_hits / self.classification_total

    @property
    def attribute_accuracy(self) -> float:
        if not self.attribute_total:
            return 0.0
        return self.attribute_hits / self.attribute_total

    @property
    def feasibility_rate(self) -> float:
        return self.feasible_hits / self.feasible_total if self.feasible_total else 0.0

    @property
    def graph_validity_rate(self) -> float:
        if not self.graph_valid_total:
            return 0.0
        return self.graph_valid_hits / self.graph_valid_total

    @property
    def offset_integrity_rate(self) -> float:
        return self.offset_hits / self.offset_total if self.offset_total else 0.0


def load_fixture(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}.")
    return value


def _find_occurrence(text: str, needle: str, occurrence: int = 0) -> tuple[int, int]:
    start = -1
    search_from = 0
    for _ in range(occurrence + 1):
        start = text.find(needle, search_from)
        if start < 0:
            raise ValueError(f"Expected text {needle!r} was not found in fixture input.")
        search_from = start + len(needle)
    return start, start + len(needle)


def expected_entity_set(case: dict[str, Any]) -> set[tuple[str, ...]]:
    text = str(case["text"])
    expected: set[tuple[str, ...]] = set()
    for item in case["expected"]:
        entity_text = str(item["text"])
        start, end = _find_occurrence(text, entity_text, int(item.get("occurrence", 0)))
        expected.add((str(item["label"]), entity_text, str(start), str(end)))
    return expected


def predicted_entity_set(result: object) -> set[tuple[str, ...]]:
    raw = result_to_dict(result)
    entities = raw.get("entities", {})
    predicted: set[tuple[str, ...]] = set()
    if not isinstance(entities, dict):
        return predicted
    for label, items in entities.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            start = item.get("start")
            end = item.get("end")
            if isinstance(text, str) and isinstance(start, int) and isinstance(end, int):
                predicted.add((str(label), text, str(start), str(end)))
    return predicted


def expected_relation_set(case: dict[str, Any]) -> set[tuple[str, ...]]:
    return {
        (str(item["type"]), str(item["head"]), str(item["tail"]))
        for item in case["expected"]
    }


def predicted_relation_set(result: Any) -> set[tuple[str, ...]]:
    predicted: set[tuple[str, ...]] = set()
    for relation in result.relations:
        head = result.entity(relation.head)
        tail = result.entity(relation.tail)
        predicted.add((str(relation.type), str(head.text), str(tail.text)))
    return predicted


def graph_is_valid(result: Any) -> bool:
    if not bool(result.feasible):
        return False
    entity_types = {entity.id: entity.type for entity in result.entities}
    endpoint_types = {
        "works_for": ("person", "organization"),
        "located_in": ("organization", "location"),
    }
    unique_heads: set[tuple[str, str]] = set()
    for relation in result.relations:
        if relation.head == relation.tail:
            return False
        if relation.head not in entity_types or relation.tail not in entity_types:
            return False
        expected_types = endpoint_types.get(str(relation.type))
        if expected_types is None:
            return False
        actual_types = (entity_types[relation.head], entity_types[relation.tail])
        if actual_types != expected_types:
            return False
        key = (str(relation.type), str(relation.head))
        if key in unique_heads:
            return False
        unique_heads.add(key)
    return True


def expected_attribute_set(case: dict[str, Any]) -> set[tuple[str, ...]]:
    return {
        (
            str(item["entity_type"]),
            str(item["text"]),
            str(item["attribute"]),
            str(item["value"]),
        )
        for item in case["expected"]
    }


def _attribute_labels(value: object) -> list[str]:
    if isinstance(value, dict):
        label = value.get("label")
        return [label] if isinstance(label, str) else []
    if isinstance(value, list):
        labels = []
        for item in value:
            if isinstance(item, dict) and isinstance(item.get("label"), str):
                labels.append(str(item["label"]))
        return labels
    return []


def predicted_attribute_set(
    result: object,
    attribute_names: set[str],
) -> set[tuple[str, ...]]:
    raw = result_to_dict(result)
    entities = raw.get("entities", {})
    predicted: set[tuple[str, ...]] = set()
    if not isinstance(entities, dict):
        return predicted
    for entity_type, items in entities.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            for attribute in attribute_names:
                for label in _attribute_labels(item.get(attribute)):
                    predicted.add((str(entity_type), str(item["text"]), attribute, label))
    return predicted


def _offsets_valid(text: str, result: object, metrics: ModelEval) -> bool:
    metrics.offset_total += 1
    try:
        assert_source_spans(text, result)
    except AssertionError:
        return False
    metrics.offset_hits += 1
    return True


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


def _guardrail_schema() -> Any:
    from gliner2.classification import ClassificationSchema
    from gliner2.classification import constraints as C

    return (
        ClassificationSchema()
        .single("safety", ["safe", "unsafe"])
        .single("harm_type", ["benign", "prompt_injection", "data_exfiltration"])
        .constrain(
            C.implies(("safety", "safe"), ("harm_type", "benign")),
            C.implies(("harm_type", "prompt_injection"), ("safety", "unsafe")),
            C.implies(("harm_type", "data_exfiltration"), ("safety", "unsafe")),
            C.excludes(("safety", "unsafe"), ("harm_type", "benign")),
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


def _attribute_schema(model: Any, attribute_group: type[Any]) -> Any:
    return (
        model.create_schema()
        .entities(
            {
                "symptom": "Symptoms or clinical findings",
                "medication": "Medication or drug names",
                "dosage": "Medication dose amounts",
            }
        )
        .entity_attributes(
            {
                "negation_status": attribute_group(
                    ["present", "negated"],
                    applies_to=["symptom"],
                    qualify_labels=True,
                ),
                "dosage_form": attribute_group(
                    ["tablet", "capsule", "liquid", "injection", "unspecified"],
                    applies_to=["medication"],
                    qualify_labels=True,
                ),
            }
        )
    )


def run_model(alias: str, device: str, fixture: dict[str, Any]) -> ModelEval:
    from gliner2 import AttributeGroup, AutoExtractor
    from gliner2.classification import ClassificationConfig, Classifier
    from gliner2.joint_ie import JointIE, JointIEConfig

    model_id = MODEL_IDS[alias]
    print(f"\nLoading {alias}: {model_id}", flush=True)
    started = time.perf_counter()
    model = AutoExtractor.from_pretrained(model_id, map_location=device)
    model.eval()
    metrics = ModelEval(
        alias=alias,
        model_id=model_id,
        load_seconds=time.perf_counter() - started,
    )

    for case in fixture["entity_cases"]:
        print(f"  entity: {case['id']}", flush=True)
        if case.get("long"):
            result = model.extract_entities_long(
                case["text"],
                case["labels"],
                chunk_size=int(case["chunk_size"]),
                chunk_overlap=int(case["chunk_overlap"]),
                include_spans=True,
                include_confidence=True,
            )
        else:
            result = model.extract_entities(
                case["text"],
                case["labels"],
                include_spans=True,
                include_confidence=True,
            )
        entity_expected = expected_entity_set(case)
        entity_predicted = predicted_entity_set(result)
        metrics.entity.add(entity_expected, entity_predicted)
        offsets_valid = _offsets_valid(str(case["text"]), result, metrics)
        metrics.records.append(
            CaseRecord(
                "entities",
                str(case["id"]),
                sorted(entity_expected),
                sorted(entity_predicted),
                entity_expected == entity_predicted,
                offsets_valid,
            )
        )

    classifier = Classifier(model, device=device).eval()
    schemas = {"routing": _routing_schema(), "guardrail": _guardrail_schema()}
    classification_config = ClassificationConfig(decoder="auto", on_infeasible="relax")
    for case in fixture["classification_cases"]:
        print(f"  classification: {case['id']}", flush=True)
        schema = schemas[str(case["schema"])]
        result = classifier.classify(case["text"], schema, config=classification_config)
        classification_expected = {
            str(key): str(value) for key, value in case["expected"].items()
        }
        classification_predicted = {
            task: str(result.value(task)) for task in classification_expected
        }
        exact = classification_expected == classification_predicted
        metrics.classification_total += 1
        metrics.classification_hits += int(exact)
        metrics.feasible_total += 1
        metrics.feasible_hits += int(bool(result.feasible))
        metrics.records.append(
            CaseRecord(
                "classification",
                str(case["id"]),
                classification_expected,
                classification_predicted,
                exact,
            )
        )

    joint = JointIE(model, device=device).eval()
    joint_schema = _joint_schema(joint)
    joint_config = JointIEConfig(optimizer="beam", beam_size=32)
    for case in fixture["relation_cases"]:
        print(f"  relation: {case['id']}", flush=True)
        result = joint.extract(case["text"], joint_schema, config=joint_config)
        relation_expected = expected_relation_set(case)
        relation_predicted = predicted_relation_set(result)
        metrics.relation.add(relation_expected, relation_predicted)
        metrics.graph_valid_total += 1
        metrics.graph_valid_hits += int(graph_is_valid(result))
        offsets_valid = _offsets_valid(str(case["text"]), result, metrics)
        metrics.records.append(
            CaseRecord(
                "relations",
                str(case["id"]),
                sorted(relation_expected),
                sorted(relation_predicted),
                relation_expected == relation_predicted,
                offsets_valid,
            )
        )

    attribute_schema = _attribute_schema(model, AttributeGroup)
    for case in fixture["attribute_cases"]:
        print(f"  attributes: {case['id']}", flush=True)
        result = model.extract(
            case["text"],
            attribute_schema,
            include_spans=True,
            include_confidence=True,
        )
        attribute_expected = expected_attribute_set(case)
        attribute_names = {item[2] for item in attribute_expected}
        attribute_predicted = predicted_attribute_set(result, attribute_names)
        metrics.attribute_total += len(attribute_expected)
        metrics.attribute_hits += len(attribute_expected & attribute_predicted)
        offsets_valid = _offsets_valid(str(case["text"]), result, metrics)
        metrics.records.append(
            CaseRecord(
                "attributes",
                str(case["id"]),
                sorted(attribute_expected),
                sorted(attribute_predicted),
                attribute_expected == attribute_predicted,
                offsets_valid,
            )
        )

    del joint, classifier, model
    gc.collect()
    return metrics


def _percent(value: float) -> str:
    return f"{value:.1%}"


def _render_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True).replace("|", "\\|")


def build_report(
    results: list[ModelEval],
    *,
    device: str,
) -> list[str]:
    lines = [
        "# GLiNER2.5 Applied Use-Case Eval",
        "",
        "Targeted source-grounded evaluation of the repository's GLiNER2.5 examples. "
        "This is not a reproduction of Fastino's 16-dataset research benchmark.",
        "",
        "## Environment",
        "",
        *environment_lines(("gliner2", "transformers", "torch", "protobuf")),
        f"- Device requested: {device}",
        f"- Fixture: `{FIXTURE_PATH.relative_to(REPO_ROOT)}`",
        "",
        "## Metrics",
        "",
        "| Model | Load (s) | Entity P/R/F1 | Relation P/R/F1 | Class accuracy | "
        "Attribute accuracy | Feasible | Graph valid | Offset integrity |",
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        entity_metric = (
            f"{_percent(result.entity.precision)} / {_percent(result.entity.recall)} / "
            f"{_percent(result.entity.f1)}"
        )
        relation_metric = (
            f"{_percent(result.relation.precision)} / "
            f"{_percent(result.relation.recall)} / {_percent(result.relation.f1)}"
        )
        lines.append(
            f"| `{result.alias}` | {result.load_seconds:.2f} | {entity_metric} | "
            f"{relation_metric} | {_percent(result.classification_accuracy)} | "
            f"{_percent(result.attribute_accuracy)} | {_percent(result.feasibility_rate)} | "
            f"{_percent(result.graph_validity_rate)} | "
            f"{_percent(result.offset_integrity_rate)} |"
        )

    for result in results:
        lines.extend(
            [
                "",
                f"## `{result.alias}` cases",
                "",
                f"Model `{result.model_id}`.",
                "",
                "| Category | Case | Expected | Predicted | Exact | Offsets |",
                "|---|---|---|---|---:|---:|",
            ]
        )
        for record in result.records:
            offsets = "n/a" if record.offsets_valid is None else (
                "pass" if record.offsets_valid else "FAIL"
            )
            lines.append(
                f"| {record.category} | `{record.case_id}` | "
                f"`{_render_json(record.expected)}` | `{_render_json(record.predicted)}` | "
                f"{'pass' if record.exact else 'FAIL'} | {offsets} |"
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Entity and relation metrics use exact label, source text, and character spans.",
            "- Classification accuracy requires the complete constrained assignment to match.",
            "- Attribute accuracy counts exact expected entity, attribute, and value matches.",
            (
                "- Graph validity checks feasibility, typed endpoints, no self-loops, "
                "and unique heads."
            ),
            (
                "- Offset integrity requires every returned span to slice back to "
                "identical source text."
            ),
            "- Quality numbers apply only to this compact fixture; tune schemas on domain data.",
        ]
    )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_IDS),
        default=list(MODEL_IDS),
        help="Model aliases to evaluate sequentially.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = normalize_device(args.device)
    fixture = load_fixture(args.fixture)
    results = [run_model(alias, device, fixture) for alias in args.models]
    lines = build_report(results, device=device)
    report_path = write_report(
        name=f"gliner25-applied-eval-{slugify(device)}",
        lines=lines,
        output_dir=args.output_dir or REPORTS_DIR,
    )
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from evals.eval_gliner25 import (
    ModelEval,
    PRFCounts,
    build_report,
    expected_entity_set,
    graph_is_valid,
    predicted_entity_set,
)
from examples.gliner25.common import (
    MODEL_IDS,
    assert_source_spans,
    normalize_device,
    resolve_model_id,
    result_to_dict,
)


class _TypedResult:
    def to_dict(self) -> dict[str, object]:
        return {"value": "ok"}


@dataclass(frozen=True)
class _Entity:
    id: str
    type: str
    text: str


@dataclass(frozen=True)
class _Relation:
    type: str
    head: str
    tail: str


class _Graph:
    def __init__(
        self,
        *,
        entities: list[_Entity],
        relations: list[_Relation],
        feasible: bool = True,
    ) -> None:
        self.entities = entities
        self.relations = relations
        self.feasible = feasible
        self._entities = {entity.id: entity for entity in entities}

    def entity(self, entity_id: str) -> _Entity:
        return self._entities[entity_id]


def test_model_aliases_and_explicit_devices() -> None:
    assert resolve_model_id("small") == MODEL_IDS["small"]
    assert resolve_model_id(MODEL_IDS["base"]) == MODEL_IDS["base"]
    assert normalize_device("gpu") == "cuda"
    assert normalize_device("cuda:2") == "cuda:2"

    with pytest.raises(ValueError, match="Unknown GLiNER2.5 model"):
        resolve_model_id("large")
    with pytest.raises(ValueError, match="Unsupported device"):
        normalize_device("tpu")


def test_result_normalization_and_recursive_span_validation() -> None:
    assert result_to_dict(_TypedResult()) == {"value": "ok"}
    text = "Alice works for Acme."
    result = {
        "entities": {
            "person": [{"text": "Alice", "start": 0, "end": 5}],
            "company": [{"text": "Acme", "start": 16, "end": 20}],
        }
    }

    assert assert_source_spans(text, result) == 2

    result["entities"]["company"][0]["text"] = "Wrong"
    with pytest.raises(AssertionError, match="Span mismatch"):
        assert_source_spans(text, result)


def test_exact_entity_scoring_includes_offsets() -> None:
    case: dict[str, Any] = {
        "text": "Apple hired Maya Chen.",
        "expected": [
            {"label": "company", "text": "Apple"},
            {"label": "person", "text": "Maya Chen"},
        ],
    }
    result = {
        "entities": {
            "company": [{"text": "Apple", "start": 0, "end": 5}],
            "person": [{"text": "Maya Chen", "start": 12, "end": 21}],
        }
    }
    expected = expected_entity_set(case)
    predicted = predicted_entity_set(result)
    counts = PRFCounts()
    counts.add(expected, predicted)

    assert expected == predicted
    assert counts.precision == 1.0
    assert counts.recall == 1.0
    assert counts.f1 == 1.0


def test_graph_validation_enforces_types_and_unique_heads() -> None:
    entities = [
        _Entity("e1", "person", "Alice"),
        _Entity("e2", "organization", "Acme"),
        _Entity("e3", "organization", "Globex"),
        _Entity("e4", "location", "Paris"),
    ]
    valid = _Graph(
        entities=entities,
        relations=[
            _Relation("works_for", "e1", "e2"),
            _Relation("located_in", "e2", "e4"),
        ],
    )
    duplicate_head = _Graph(
        entities=entities,
        relations=[
            _Relation("works_for", "e1", "e2"),
            _Relation("works_for", "e1", "e3"),
        ],
    )

    assert graph_is_valid(valid) is True
    assert graph_is_valid(duplicate_head) is False


def test_eval_report_contains_metrics_and_case_details() -> None:
    result = ModelEval(alias="small", model_id=MODEL_IDS["small"], load_seconds=1.25)
    result.entity.add({("person", "Alice", "0", "5")}, {("person", "Alice", "0", "5")})
    result.relation.add(
        {("works_for", "Alice", "Acme")},
        {("works_for", "Alice", "Acme")},
    )
    result.classification_hits = result.classification_total = 1
    result.attribute_hits = result.attribute_total = 1
    result.feasible_hits = result.feasible_total = 1
    result.graph_valid_hits = result.graph_valid_total = 1
    result.offset_hits = result.offset_total = 1

    report = "\n".join(build_report([result], device="cpu"))

    assert "# GLiNER2.5 Applied Use-Case Eval" in report
    assert "`small`" in report
    assert "100.0% / 100.0% / 100.0%" in report
    assert "not a reproduction" in report

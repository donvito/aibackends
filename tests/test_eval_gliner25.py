"""Unit tests for GLiNER 2.5 eval scoring (no Hub download)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from aibackends.schemas.extraction import (
    EntityExtraction,
    ExtractedEntity,
    GraphExtraction,
    GraphRelation,
    RecordExtraction,
    SchemaClassification,
    SpanAttribute,
    TaskClassification,
)
from aibackends.schemas.pii import RedactedText

_EVALS = Path(__file__).resolve().parents[1] / "evals"
if str(_EVALS) not in sys.path:
    sys.path.insert(0, str(_EVALS))

eval_gliner25: ModuleType = importlib.import_module("eval_gliner25")


def _case(**kwargs: Any) -> Any:
    return eval_gliner25.EvalCase(name="case", use_case="test", run=lambda: None, **kwargs)


def test_contains_hit_allows_extra_predictions() -> None:
    precision, recall, ok = eval_gliner25._contains_hit(
        {"safety:unsafe", "harm_type:prompt_injection"},
        {"safety:unsafe"},
    )
    assert ok is True
    assert recall == 1.0
    assert precision == 0.5


def test_contains_hit_fails_when_gold_is_missing() -> None:
    _precision, recall, ok = eval_gliner25._contains_hit(
        {"person:ada lovelace"},
        {"person:ada lovelace", "location:london"},
    )
    assert ok is False
    assert recall == 0.5


def test_score_classification_requires_labels_and_feasible() -> None:
    case = _case(
        expected_labels={"intent": "delete", "destination": "file_tool"},
        require_feasible=True,
    )
    predicted = SchemaClassification(
        text="delete the file",
        tasks={
            "intent": TaskClassification(task="intent", value="delete"),
            "destination": TaskClassification(task="destination", value="file_tool"),
        },
        feasible=True,
        backend_used="gliner25",
        model_id="fastino/gliner2.5-small-v1",
    )
    passed, precision, recall, _detail = eval_gliner25.score_case(case, predicted)
    assert passed is True
    assert precision == 1.0
    assert recall == 1.0

    infeasible = predicted.model_copy(update={"feasible": False})
    passed, _precision, _recall, _detail = eval_gliner25.score_case(case, infeasible)
    assert passed is False


def test_score_entities_and_attributes() -> None:
    gold = eval_gliner25.GoldEntity
    case = _case(
        expected_entities=(
            gold("symptom", "fever", ("negation", "negated")),
            gold("medication", "amoxicillin"),
        )
    )
    predicted = EntityExtraction(
        text="denies fever, started amoxicillin 500mg capsules",
        entities=[
            ExtractedEntity(
                entity_type="symptom",
                text="fever",
                attributes={
                    "negation": SpanAttribute(name="negation", label="negated"),
                },
            ),
            ExtractedEntity(entity_type="medication", text="amoxicillin 500mg capsules"),
            ExtractedEntity(entity_type="symptom", text="cough"),
        ],
        backend_used="gliner25",
        model_id="fastino/gliner2.5-small-v1",
    )
    passed, precision, recall, _detail = eval_gliner25.score_case(case, predicted)
    assert passed is True
    assert recall == 1.0
    assert precision < 1.0


def test_score_graph_triples() -> None:
    case = _case(
        expected_triples=(("Ada Lovelace", "works_for", "Fastino Labs"),),
        require_feasible=True,
    )
    predicted = GraphExtraction(
        text="Ada Lovelace works at Fastino Labs.",
        relations=[
            GraphRelation(
                relation_type="works_for",
                head_id="e1",
                tail_id="e2",
                head_text="Ada Lovelace",
                tail_text="Fastino Labs",
            )
        ],
        feasible=True,
        backend_used="gliner25",
        model_id="fastino/gliner2.5-small-v1",
    )
    passed, precision, recall, _detail = eval_gliner25.score_case(case, predicted)
    assert passed is True
    assert precision == 1.0
    assert recall == 1.0


def test_score_redaction() -> None:
    case = _case(require_redacted=("landlord@sampledomain.test",))
    predicted = RedactedText(
        original_text="Contact landlord@sampledomain.test",
        redacted_text="Contact [EMAIL]",
        entities_found=[],
        redaction_map={},
        backend_used="gliner25",
    )
    passed, precision, recall, _detail = eval_gliner25.score_case(case, predicted)
    assert passed is True
    assert precision == 1.0
    assert recall == 1.0

    visible = predicted.model_copy(
        update={"redacted_text": "Contact landlord@sampledomain.test"}
    )
    passed, _precision, _recall, _detail = eval_gliner25.score_case(case, visible)
    assert passed is False


def test_score_record_fields() -> None:
    case = _case(expected_fields={"provider": "Northwind Analytics LLC"})
    predicted = RecordExtraction(
        text="Northwind Analytics LLC invoices Contoso.",
        records={
            "agreement": [
                {
                    "provider": {"text": "Northwind Analytics LLC"},
                    "customer": {"text": "Contoso Retail Inc."},
                }
            ]
        },
        backend_used="gliner25",
        model_id="fastino/gliner2.5-small-v1",
    )
    passed, precision, recall, _detail = eval_gliner25.score_case(case, predicted)
    assert passed is True
    assert recall == 1.0
    assert precision == 0.5


def test_build_report_lines_include_pass_rate() -> None:
    args = type("Args", (), {"model": "gliner25-small", "device": "cpu"})()
    results = [
        eval_gliner25.CaseResult(
            name="route",
            use_case="agent-routing",
            passed=True,
            precision=1.0,
            recall=1.0,
            detail="ok",
            elapsed_ms=20.0,
        )
    ]
    lines = eval_gliner25.build_report_lines(args, results)
    joined = "\n".join(lines)
    assert "Exact case pass rate | 1/1 (100%)" in joined
    assert "agent-routing | 1/1 | 1.00" in joined

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from aibackends.backends.extraction import get_extraction_backend
from aibackends.core.exceptions import TaskExecutionError
from aibackends.tasks import (
    ExtractEntitiesTask,
    classify_text,
    extract_entities,
    extract_entities_batch,
    extract_graph,
)

gliner25_module = importlib.import_module("aibackends.backends.extraction.gliner25")

SMALL_ID = "fastino/gliner2.5-small-v1"
BASE_ID = "fastino/gliner2.5-base-v1"


class _FakeSchema:
    def __init__(self) -> None:
        self.entity_types: Any = None
        self.attribute_groups: Any = None

    def entities(self, entity_types: Any) -> _FakeSchema:
        self.entity_types = entity_types
        return self

    def entity_attributes(self, groups: Any) -> _FakeSchema:
        self.attribute_groups = groups
        return self


class _FakeExtractor:
    def __init__(self, batch_results: list[dict[str, Any]] | None = None) -> None:
        self.batch_results = batch_results or []
        self.calls: list[dict[str, Any]] = []

    def create_schema(self) -> _FakeSchema:
        return _FakeSchema()

    def batch_extract(self, texts: list[str], schema: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"method": "batch_extract", "texts": list(texts), **kwargs})
        return [dict(result) for result in self.batch_results]

    def batch_extract_long(
        self, texts: list[str], schema: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls.append({"method": "batch_extract_long", "texts": list(texts), **kwargs})
        return [dict(result) for result in self.batch_results]


@dataclass
class _FakeAttributeGroup:
    labels: list[str]
    multi_label: bool = False
    threshold: float = 0.5
    applies_to: list[str] | None = None
    qualify_labels: bool = False


class _FakeClassificationSchema:
    def __init__(self) -> None:
        self.tasks: list[tuple[str, str, Any, dict[str, Any]]] = []
        self.constraints: list[Any] = []

    def single(self, name: str, labels: Any, **kwargs: Any) -> _FakeClassificationSchema:
        self.tasks.append(("single", name, labels, kwargs))
        return self

    def multi(self, name: str, labels: Any, **kwargs: Any) -> _FakeClassificationSchema:
        self.tasks.append(("multi", name, labels, kwargs))
        return self

    def constrain(self, *expressions: Any) -> _FakeClassificationSchema:
        self.constraints.extend(expressions)
        return self


@dataclass
class _FakeClassificationConfig:
    batch_size: int = 8


class _FakeClassifier:
    def __init__(self, batch_results: list[Any] | None = None) -> None:
        self.batch_results = batch_results or []
        self.calls: list[dict[str, Any]] = []

    def batch_classify(self, texts: list[str], schema: Any, *, config: Any) -> list[Any]:
        self.calls.append({"texts": list(texts), "schema": schema, "config": config})
        return list(self.batch_results)


class _FakeJointSchema:
    def __init__(self) -> None:
        self.entity_types: Any = None
        self.relations: list[tuple[str, Any, Any, dict[str, Any]]] = []
        self.self_loops_forbidden = False

    def entities(self, entity_types: Any) -> _FakeJointSchema:
        self.entity_types = entity_types
        return self

    def relation(self, name: str, head: Any, tail: Any, **kwargs: Any) -> _FakeJointSchema:
        self.relations.append((name, head, tail, kwargs))
        return self

    def no_self_loops(self) -> _FakeJointSchema:
        self.self_loops_forbidden = True
        return self


@dataclass
class _FakeJointIEConfig:
    optimizer: str = "beam"
    beam_size: int = 32


@dataclass
class _FakeJointResult:
    entities: list[Any] = field(default_factory=list)
    relations: list[Any] = field(default_factory=list)
    feasible: bool = True


class _FakeJoint:
    def __init__(self, result: _FakeJointResult | None = None) -> None:
        self.result = result or _FakeJointResult()
        self.calls: list[dict[str, Any]] = []

    def create_schema(self) -> _FakeJointSchema:
        return _FakeJointSchema()

    def extract(self, text: str, schema: Any, *, config: Any) -> _FakeJointResult:
        self.calls.append({"text": text, "schema": schema, "config": config})
        return self.result


@pytest.fixture(autouse=True)
def _reset_gliner25_cache() -> Iterator[None]:
    gliner25_module.clear_model_cache()
    yield
    gliner25_module.clear_model_cache()


@pytest.fixture()
def _fake_gliner2_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_package = ModuleType("gliner2")
    fake_package.AttributeGroup = _FakeAttributeGroup  # type: ignore[attr-defined]

    fake_constraints = ModuleType("gliner2.classification.constraints")
    for kind in ("implies", "excludes", "iff"):
        setattr(fake_constraints, kind, lambda a, b, _kind=kind: (_kind, a, b))

    fake_classification = ModuleType("gliner2.classification")
    fake_classification.ClassificationSchema = (  # type: ignore[attr-defined]
        _FakeClassificationSchema
    )
    fake_classification.ClassificationConfig = (  # type: ignore[attr-defined]
        _FakeClassificationConfig
    )
    fake_classification.constraints = fake_constraints  # type: ignore[attr-defined]

    fake_joint_ie = ModuleType("gliner2.joint_ie")
    fake_joint_ie.JointIEConfig = _FakeJointIEConfig  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gliner2", fake_package)
    monkeypatch.setitem(sys.modules, "gliner2.classification", fake_classification)
    monkeypatch.setitem(
        sys.modules, "gliner2.classification.constraints", fake_constraints
    )
    monkeypatch.setitem(sys.modules, "gliner2.joint_ie", fake_joint_ie)


def _install_fake_extractor(fake: _FakeExtractor, *, model_id: str = BASE_ID) -> _FakeExtractor:
    gliner25_module._MODEL_CACHE[(model_id, "cpu")] = fake
    return fake


def _install_fake_classifier(fake: _FakeClassifier, *, model_id: str = BASE_ID) -> _FakeClassifier:
    gliner25_module._CLASSIFIER_CACHE[(model_id, "cpu")] = fake
    return fake


def _install_fake_joint(fake: _FakeJoint, *, model_id: str = BASE_ID) -> _FakeJoint:
    gliner25_module._JOINT_CACHE[(model_id, "cpu")] = fake
    return fake


def test_resolve_model_id_maps_variants_and_passes_repo_ids() -> None:
    assert gliner25_module.resolve_model_id(None) == BASE_ID
    assert gliner25_module.resolve_model_id("small") == SMALL_ID
    assert gliner25_module.resolve_model_id("multi") == "fastino/gliner2.5-multi-v1"
    assert gliner25_module.resolve_model_id("acme/custom-model") == "acme/custom-model"
    with pytest.raises(ValueError, match="Unknown GLiNER2.5 model"):
        gliner25_module.resolve_model_id("tiny")


def test_extract_entities_parses_spans_and_attributes() -> None:
    fake = _install_fake_extractor(
        _FakeExtractor(
            batch_results=[
                {
                    "entities": {
                        "person": [
                            {
                                "text": "Alice",
                                "start": 0,
                                "end": 5,
                                "confidence": 0.97,
                                "sentiment": {"label": "positive", "confidence": 0.88},
                            }
                        ],
                        "company": [{"text": "Acme", "start": 16, "end": 20}],
                    }
                }
            ]
        )
    )

    result = extract_entities(
        "Alice works for Acme.",
        labels=["person", "company"],
        threshold=0.6,
    )

    assert result.backend_used == "gliner2.5"
    assert result.model_id == BASE_ID
    assert [entity.label for entity in result.entities] == ["person", "company"]
    person = result.entities[0]
    assert (person.start, person.end, person.confidence) == (0, 5, 0.97)
    assert person.attributes["sentiment"].labels == ["positive"]
    assert person.attributes["sentiment"].confidences == {"positive": 0.88}
    assert fake.calls[0]["method"] == "batch_extract"
    assert fake.calls[0]["threshold"] == 0.6


def test_extract_entities_long_document_uses_chunked_path() -> None:
    fake = _install_fake_extractor(
        _FakeExtractor(batch_results=[{"entities": {"person": []}}])
    )

    extract_entities(
        "long text",
        labels=["person"],
        long_document=True,
        chunk_size=128,
        chunk_overlap=32,
    )

    assert fake.calls[0]["method"] == "batch_extract_long"
    assert fake.calls[0]["chunk_size"] == 128
    assert fake.calls[0]["chunk_overlap"] == 32


def test_extract_entities_batch_uses_native_batching() -> None:
    fake = _install_fake_extractor(
        _FakeExtractor(
            batch_results=[
                {"entities": {"person": [{"text": "Alice", "start": 0, "end": 5}]}},
                {"entities": {"person": []}},
            ]
        )
    )

    results = extract_entities_batch(
        ["Alice.", "Nothing here."],
        labels=["person"],
        batch_size=2,
    )

    assert len(results) == 2
    assert results[0].entities[0].text == "Alice"
    assert results[1].entities == []
    assert fake.calls[0]["batch_size"] == 2


def test_extract_entities_rejects_invalid_output() -> None:
    _install_fake_extractor(_FakeExtractor(batch_results=[{"unexpected": True}]))

    with pytest.raises(TaskExecutionError, match="invalid entity extraction"):
        extract_entities("Alice.", labels=["person"])


def test_extract_entities_validates_options_before_loading() -> None:
    backend = get_extraction_backend("gliner25")

    with pytest.raises(ValueError, match="threshold"):
        backend.extract_entities("hello", labels=["person"], threshold=1.4)
    with pytest.raises(ValueError, match="chunk_overlap"):
        backend.extract_entities(
            "hello", labels=["person"], chunk_size=64, chunk_overlap=64
        )
    with pytest.raises(ValueError, match="labels must not be empty"):
        backend.extract_entities("hello", labels=[])
    with pytest.raises(ValueError, match="Unsupported GLiNER2.5 device"):
        backend.load(device="tpu")


def test_classify_text_builds_constrained_schema(
    _fake_gliner2_modules: None,
) -> None:
    fake = _install_fake_classifier(
        _FakeClassifier(
            batch_results=[
                SimpleNamespace(
                    tasks={
                        "intent": SimpleNamespace(
                            labels=("delete",),
                            probabilities={"read": 0.02, "delete": 0.93},
                            confidence=0.93,
                            exclusive=True,
                        ),
                        "effects": SimpleNamespace(
                            labels=("delete",),
                            probabilities={"delete": 0.88},
                            confidence=0.88,
                            exclusive=False,
                        ),
                    },
                    feasible=True,
                )
            ]
        )
    )

    result = classify_text(
        "Delete the temporary file",
        tasks={
            "intent": {"labels": ["read", "write", "delete"]},
            "effects": {"labels": ["read_only", "delete"], "multi_label": True},
        },
        constraints=[
            {"kind": "implies", "when": ["intent", "delete"], "then": ["effects", "delete"]},
        ],
    )

    assert result.value("intent") == "delete"
    assert result.values("effects") == ["delete"]
    assert result.feasible is True
    assert result.constrained is True
    assert result.tasks["effects"].multi_label is True
    schema = fake.calls[0]["schema"]
    assert ("single", "intent", ["read", "write", "delete"], {}) in schema.tasks
    assert ("multi", "effects", ["read_only", "delete"], {}) in schema.tasks
    assert schema.constraints == [
        ("implies", ("intent", "delete"), ("effects", "delete"))
    ]


def test_classify_text_rejects_unknown_constraint_kind(
    _fake_gliner2_modules: None,
) -> None:
    _install_fake_classifier(_FakeClassifier())

    with pytest.raises(ValueError, match="Unknown constraint kind"):
        classify_text(
            "hello",
            tasks={"intent": ["a", "b"]},
            constraints=[{"kind": "banns", "when": ["intent", "a"], "then": ["intent", "b"]}],
        )
    with pytest.raises(ValueError, match="must be a \\[task, label\\] pair"):
        classify_text(
            "hello",
            tasks={"intent": ["a", "b"]},
            constraints=[{"kind": "implies", "when": "intent", "then": ["intent", "b"]}],
        )


def test_classify_text_accepts_plain_labels_shorthand(
    _fake_gliner2_modules: None,
) -> None:
    fake = _install_fake_classifier(
        _FakeClassifier(
            batch_results=[
                SimpleNamespace(
                    tasks={
                        "label": SimpleNamespace(
                            labels=("billing",),
                            probabilities={"billing": 0.9},
                            confidence=0.9,
                            exclusive=True,
                        )
                    },
                    feasible=True,
                )
            ]
        )
    )

    result = classify_text("Refund my card", labels=["billing", "bug", "feature"])

    assert result.value("label") == "billing"
    schema = fake.calls[0]["schema"]
    assert schema.tasks == [("single", "label", ["billing", "bug", "feature"], {})]


def test_extract_graph_builds_joint_schema_and_parses_result(
    _fake_gliner2_modules: None,
) -> None:
    fake = _install_fake_joint(
        _FakeJoint(
            result=_FakeJointResult(
                entities=[
                    SimpleNamespace(
                        id="e1", type="person", text="Alice", start=0, end=5, confidence=0.9
                    ),
                    SimpleNamespace(
                        id="e2", type="organization", text="Acme", start=16, end=20,
                        confidence=0.8,
                    ),
                ],
                relations=[
                    SimpleNamespace(type="works_for", head="e1", tail="e2", confidence=0.7)
                ],
                feasible=True,
            )
        )
    )

    graph = extract_graph(
        "Alice works for Acme.",
        entities=["person", "organization"],
        relations=[
            {"name": "works_for", "head": "person", "tail": "organization",
             "unique_head": True},
        ],
    )

    assert graph.feasible is True
    assert graph.triples() == [("Alice", "works_for", "Acme")]
    assert graph.entity("e2").type == "organization"
    schema = fake.calls[0]["schema"]
    assert schema.entity_types == ["person", "organization"]
    assert schema.relations == [
        ("works_for", "person", "organization", {"unique_head": True})
    ]
    assert schema.self_loops_forbidden is True
    assert fake.calls[0]["config"].beam_size == 32


def test_extract_graph_requires_relations(_fake_gliner2_modules: None) -> None:
    _install_fake_joint(_FakeJoint())

    with pytest.raises(ValueError, match="relations must not be empty"):
        extract_graph("hello", entities=["person"], relations=[])
    with pytest.raises(ValueError, match="'name', 'head', and 'tail'"):
        extract_graph("hello", entities=["person"], relations=[{"name": "x"}])


def test_task_class_coerces_model_strings() -> None:
    _install_fake_extractor(
        _FakeExtractor(batch_results=[{"entities": {"person": []}}]),
        model_id=SMALL_ID,
    )

    task = ExtractEntitiesTask(model="small", device="cpu")
    result = task.run("Nothing here.", labels=["person"])

    assert result.model_id == SMALL_ID


def test_extract_entities_rejects_unknown_backend() -> None:
    with pytest.raises(TaskExecutionError, match="Unsupported extraction backend"):
        extract_entities("hello", labels=["person"], backend="unknown")

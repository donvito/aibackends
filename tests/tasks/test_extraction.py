from __future__ import annotations

import importlib
from collections.abc import Iterator
from typing import Any

import pytest

from aibackends.backends.extraction import get_extraction_backend, list_extraction_backends
from aibackends.backends.pii import get_pii_backend
from aibackends.tasks import (
    ClassifySchemaTask,
    ExtractEntitiesTask,
    ExtractGraphTask,
    classify_schema,
    extract_entities,
    extract_entities_batch,
    extract_graph,
    extract_records,
    redact_pii,
)

gliner25_module = importlib.import_module("aibackends.backends.extraction.gliner25")


class _FakeSchema:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def entities(self, labels: Any) -> _FakeSchema:
        self.calls.append(("entities", labels))
        return self

    def entity_attributes(self, groups: Any) -> _FakeSchema:
        self.calls.append(("entity_attributes", groups))
        return self

    def relation(self, *args: Any, **kwargs: Any) -> _FakeSchema:
        self.calls.append(("relation", args, kwargs))
        return self

    def no_self_loops(self) -> _FakeSchema:
        self.calls.append(("no_self_loops", None))
        return self


class _FakeExtractor:
    def __init__(
        self,
        *,
        entities: dict[str, Any] | None = None,
        records: dict[str, Any] | None = None,
        classification: dict[str, Any] | None = None,
    ) -> None:
        self.entities = entities or {"entities": {}}
        self.records = records or {}
        self.classification = classification or {}
        self.calls: list[dict[str, Any]] = []
        self.schema = _FakeSchema()

    def eval(self) -> _FakeExtractor:
        return self

    def create_schema(self) -> _FakeSchema:
        return self.schema

    def extract_entities(self, text: str, labels: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            {"method": "extract_entities", "text": text, "labels": labels, **kwargs}
        )
        return dict(self.entities)

    def extract_entities_long(self, text: str, labels: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "extract_entities_long",
                "text": text,
                "labels": labels,
                **kwargs,
            }
        )
        return dict(self.entities)

    def batch_extract_entities(
        self, texts: list[str], labels: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "method": "batch_extract_entities",
                "texts": list(texts),
                "labels": labels,
                **kwargs,
            }
        )
        return [dict(self.entities) for _ in texts]

    def batch_extract_entities_long(
        self, texts: list[str], labels: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "method": "batch_extract_entities_long",
                "texts": list(texts),
                "labels": labels,
                **kwargs,
            }
        )
        return [dict(self.entities) for _ in texts]

    def batch_extract(self, texts: list[str], schema: Any, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "method": "batch_extract",
                "texts": list(texts),
                "schema": schema,
                **kwargs,
            }
        )
        return [dict(self.entities) for _ in texts]

    def batch_extract_long(
        self, texts: list[str], schema: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls.append(
            {
                "method": "batch_extract_long",
                "texts": list(texts),
                "schema": schema,
                **kwargs,
            }
        )
        return [dict(self.entities) for _ in texts]

    def extract(self, text: str, schema: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": "extract", "text": text, "schema": schema, **kwargs})
        return dict(self.entities)

    def extract_long(self, text: str, schema: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            {"method": "extract_long", "text": text, "schema": schema, **kwargs}
        )
        return dict(self.entities)

    def extract_json(self, text: str, structures: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "extract_json",
                "text": text,
                "structures": structures,
                **kwargs,
            }
        )
        return dict(self.records)

    def extract_json_long(self, text: str, structures: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "extract_json_long",
                "text": text,
                "structures": structures,
                **kwargs,
            }
        )
        return dict(self.records)

    def classify_text(self, text: str, schema: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(
            {"method": "classify_text", "text": text, "schema": schema, **kwargs}
        )
        return dict(self.classification)


class _FakeClassResult:
    def __init__(self, values: dict[str, Any], *, feasible: bool = True) -> None:
        self._values = values
        self.feasible = feasible

    def value(self, task: str) -> Any:
        return self._values[task]

    def confidence(self, task: str) -> float:
        return 0.91

    def probabilities(self, task: str) -> dict[str, float]:
        return {"delete": 0.91}


class _FakeClassifier:
    def __init__(self, result: _FakeClassResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def classify(self, text: str, schema: Any, config: Any = None) -> _FakeClassResult:
        self.calls.append({"text": text, "schema": schema, "config": config})
        return self.result


class _FakeGraphEntity:
    def __init__(
        self,
        entity_id: str,
        entity_type: str,
        text: str,
        start: int,
        end: int,
        confidence: float = 0.9,
    ) -> None:
        self.id = entity_id
        self.type = entity_type
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence


class _FakeGraphRelation:
    def __init__(
        self,
        relation_type: str,
        head: str,
        tail: str,
        confidence: float = 0.8,
    ) -> None:
        self.type = relation_type
        self.head = head
        self.tail = tail
        self.confidence = confidence


class _FakeGraphResult:
    def __init__(
        self,
        entities: list[_FakeGraphEntity],
        relations: list[_FakeGraphRelation],
        *,
        feasible: bool = True,
    ) -> None:
        self.entities = entities
        self.relations = relations
        self.feasible = feasible
        self._by_id = {entity.id: entity for entity in entities}

    def entity(self, entity_id: str) -> _FakeGraphEntity:
        return self._by_id[entity_id]


class _FakeJoint:
    def __init__(self, result: _FakeGraphResult) -> None:
        self.result = result
        self.schema = _FakeSchema()
        self.calls: list[dict[str, Any]] = []

    def create_schema(self) -> _FakeSchema:
        return self.schema

    def extract(self, text: str, schema: Any, config: Any = None) -> _FakeGraphResult:
        self.calls.append({"method": "extract", "text": text, "schema": schema})
        return self.result

    def extract_long(self, text: str, schema: Any, **kwargs: Any) -> _FakeGraphResult:
        self.calls.append(
            {"method": "extract_long", "text": text, "schema": schema, **kwargs}
        )
        return self.result


@pytest.fixture(autouse=True)
def _reset_gliner25_cache() -> Iterator[None]:
    gliner25_module.clear_model_cache()
    yield
    gliner25_module.clear_model_cache()


def _cache_key(model: str = "gliner25-small", device: str = "cpu") -> tuple[str, str]:
    return (gliner25_module.resolve_gliner25_model_id(model), device)


def _install_extractor(fake: _FakeExtractor, *, model: str = "gliner25-small") -> _FakeExtractor:
    gliner25_module._MODEL_CACHE[_cache_key(model)] = fake
    return fake


def test_extraction_backend_is_discoverable() -> None:
    backend = get_extraction_backend("gliner-2.5")
    assert backend.name == "gliner25"
    assert "gliner25" in list_extraction_backends()


def test_gliner25_pii_backend_is_discovered() -> None:
    backend = get_pii_backend("gliner25")
    assert backend.name == "gliner25"
    assert backend.supports_custom_labels is True
    assert backend.model_id == gliner25_module.DEFAULT_GLINER25_MODEL


def test_resolve_gliner25_aliases() -> None:
    assert (
        gliner25_module.resolve_gliner25_model_id("gliner25-base")
        == "fastino/gliner2.5-base-v1"
    )
    assert gliner25_module.resolve_gliner25_model_id(None) == (
        gliner25_module.DEFAULT_GLINER25_MODEL
    )


def test_normalize_device_aliases() -> None:
    assert gliner25_module.normalize_device("gpu") == "cuda"
    assert gliner25_module.normalize_device("cuda:1") == "cuda:1"
    with pytest.raises(ValueError, match="Unsupported GLiNER 2.5 device"):
        gliner25_module.normalize_device("tpu")


def test_extract_entities_maps_spans() -> None:
    text = "Ada Lovelace wrote to Charles Babbage in London."
    _install_extractor(
        _FakeExtractor(
            entities={
                "entities": {
                    "person": [
                        {"text": "Ada Lovelace", "start": 0, "end": 12, "confidence": 0.96}
                    ],
                    "location": [
                        {"text": "London", "start": 41, "end": 47, "confidence": 0.91}
                    ],
                }
            }
        )
    )

    result = extract_entities(text, labels=["person", "location"])

    assert result.backend_used == "gliner25"
    assert result.model_id == gliner25_module.DEFAULT_GLINER25_MODEL
    assert [entity.text for entity in result.entities] == ["Ada Lovelace", "London"]
    assert text[result.entities[0].start : result.entities[0].end] == "Ada Lovelace"


def test_extract_entities_long_uses_chunking_api() -> None:
    fake = _install_extractor(
        _FakeExtractor(entities={"entities": {"person": [{"text": "Ada", "start": 0, "end": 3}]}})
    )
    extract_entities("Ada wrote notes.", labels=["person"], long=True, chunk_size=128)

    assert fake.calls[0]["method"] == "extract_entities_long"
    assert fake.calls[0]["chunk_size"] == 128


def test_extract_entities_batch_uses_native_batch_api() -> None:
    fake = _install_extractor(
        _FakeExtractor(
            entities={
                "entities": {
                    "person": [
                        {"text": "Ada Lovelace", "start": 0, "end": 12, "confidence": 0.95}
                    ]
                }
            }
        )
    )
    texts = [
        "Ada Lovelace wrote notes.",
        "Ada Lovelace lives in London.",
    ]
    results = extract_entities_batch(texts, labels=["person"], batch_size=2)

    assert fake.calls[0]["method"] == "batch_extract_entities"
    assert fake.calls[0]["batch_size"] == 2
    assert [item.text for item in results] == texts
    assert results[0].entities[0].text == "Ada Lovelace"


def test_extract_entities_batch_long_uses_chunking_api() -> None:
    fake = _install_extractor(
        _FakeExtractor(entities={"entities": {"person": [{"text": "Ada", "start": 0, "end": 3}]}})
    )
    extract_entities_batch(
        ["Ada wrote notes."],
        labels=["person"],
        long=True,
        chunk_size=128,
        batch_size=4,
    )

    assert fake.calls[0]["method"] == "batch_extract_entities_long"
    assert fake.calls[0]["chunk_size"] == 128
    assert fake.calls[0]["batch_size"] == 4


def test_extract_entities_batch_empty_returns_no_calls() -> None:
    fake = _install_extractor(_FakeExtractor())
    assert extract_entities_batch([], labels=["person"]) == []
    assert fake.calls == []


def test_extract_entities_with_attributes_uses_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _install_extractor(
        _FakeExtractor(
            entities={
                "entities": {
                    "symptom": [
                        {
                            "text": "fever",
                            "start": 14,
                            "end": 19,
                            "confidence": 0.88,
                            "negation": {"label": "negated", "confidence": 0.84},
                        }
                    ]
                }
            }
        )
    )
    monkeypatch.setattr(
        gliner25_module,
        "_build_attribute_groups",
        lambda attributes: dict(attributes),
    )
    text = "Patient denies fever today."
    result = extract_entities(
        text,
        labels=["symptom"],
        attributes={
            "negation": {
                "labels": ["affirmed", "negated"],
                "applies_to": ["symptom"],
            }
        },
    )

    assert fake.calls[0]["method"] == "extract"
    assert result.entities[0].attributes["negation"].label == "negated"


def test_extract_records_returns_json_fields() -> None:
    _install_extractor(
        _FakeExtractor(
            records={
                "product": [
                    {
                        "name": {"text": "iPhone 15", "start": 4, "end": 13},
                        "price": "$999",
                    }
                ]
            }
        )
    )
    result = extract_records(
        "The iPhone 15 costs $999.",
        schema={"product": ["name::str", "price::str"]},
    )
    assert result.records["product"][0]["price"] == "$999"


def test_classify_schema_unconstrained() -> None:
    fake = _install_extractor(
        _FakeExtractor(classification={"intent": {"label": "delete", "confidence": 0.93}})
    )
    result = classify_schema(
        "Delete the temporary file",
        tasks={"intent": ["read", "write", "delete"]},
    )
    assert result.tasks["intent"].value == "delete"
    assert result.feasible is True
    assert fake.calls[0]["method"] == "classify_text"


def test_classify_schema_constrained_uses_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractor = _install_extractor(_FakeExtractor())
    classifier = _FakeClassifier(
        _FakeClassResult({"intent": "delete", "effects": ["delete"]})
    )
    gliner25_module._CLASSIFIER_CACHE[_cache_key()] = classifier
    monkeypatch.setattr(
        gliner25_module,
        "_build_classification_schema",
        lambda tasks, constraints: {"tasks": dict(tasks), "constraints": list(constraints)},
    )

    result = classify_schema(
        "Delete the temporary file from /tmp",
        tasks={
            "intent": {"labels": ["read", "write", "delete"]},
            "effects": {
                "labels": ["read_only", "create", "modify", "delete"],
                "multi_label": True,
                "min_labels": 1,
            },
        },
        constraints=[
            {"type": "implies", "if": ["intent", "delete"], "then": ["effects", "delete"]},
            {"type": "excludes", "left": ["intent", "read"], "right": ["effects", "delete"]},
        ],
    )

    assert result.tasks["intent"].value == "delete"
    assert result.tasks["effects"].value == ["delete"]
    assert result.feasible is True
    assert classifier.calls
    assert extractor.calls == []


def test_extract_graph_maps_typed_relations() -> None:
    text = "Tim Cook leads Apple in Cupertino."
    _install_extractor(_FakeExtractor())
    entities = [
        _FakeGraphEntity("e1", "person", "Tim Cook", 0, 8),
        _FakeGraphEntity("e2", "organization", "Apple", 15, 20),
        _FakeGraphEntity("e3", "location", "Cupertino", 24, 33),
    ]
    relations = [
        _FakeGraphRelation("works_for", "e1", "e2"),
        _FakeGraphRelation("located_in", "e2", "e3"),
    ]
    gliner25_module._JOINT_CACHE[_cache_key()] = _FakeJoint(
        _FakeGraphResult(entities, relations)
    )

    result = extract_graph(
        text,
        entities=["person", "organization", "location"],
        relations=[
            {"name": "works_for", "head": "person", "tail": "organization", "unique_head": True},
            {"name": "located_in", "head": "organization", "tail": "location"},
        ],
    )

    assert result.feasible is True
    assert result.relations[0].head_text == "Tim Cook"
    assert result.relations[0].tail_text == "Apple"
    assert text[result.entities[0].start : result.entities[0].end] == "Tim Cook"


def test_redact_pii_gliner25_uses_shared_extractor() -> None:
    text = "Email ada@example.test or call +1 555 0100."
    email = "ada@example.test"
    phone = "+1 555 0100"
    start_email = text.index(email)
    start_phone = text.index(phone)
    _install_extractor(
        _FakeExtractor(
            entities={
                "entities": {
                    "email": [
                        {
                            "text": email,
                            "start": start_email,
                            "end": start_email + len(email),
                        }
                    ],
                    "phone_number": [
                        {
                            "text": phone,
                            "start": start_phone,
                            "end": start_phone + len(phone),
                        }
                    ],
                }
            }
        )
    )

    result = redact_pii(text, backend="gliner25", labels=["email", "phone_number"])

    assert result.backend_used == "gliner25"
    assert email not in result.redacted_text
    assert phone not in result.redacted_text
    assert len(result.entities_found) == 2


def test_extract_entities_task_requires_labels() -> None:
    with pytest.raises(TypeError, match="requires labels"):
        ExtractEntitiesTask().run("Ada wrote notes.")


def test_classify_schema_task_requires_tasks() -> None:
    with pytest.raises(TypeError, match="requires a tasks mapping"):
        ClassifySchemaTask().run("Delete the file")


def test_extract_graph_task_requires_schema() -> None:
    with pytest.raises(TypeError, match="requires entities"):
        ExtractGraphTask().run("Ada works at Acme.")

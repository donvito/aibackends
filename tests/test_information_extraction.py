from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from aibackends.backends.information_extraction import (
    get_information_extraction_backend,
    list_information_extraction_backends,
)
from aibackends.backends.information_extraction.gliner25 import (
    GLINER25_MODEL_IDS,
    GLiNER25Backend,
    clear_gliner25_model_cache,
    load_gliner25_model,
)
from aibackends.tasks import information_extraction as task_module

gliner25_module = importlib.import_module(
    "aibackends.backends.information_extraction.gliner25"
)


class _FakeExtractor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, Any]]] = []
        self.evaluated = False

    def eval(self) -> _FakeExtractor:
        self.evaluated = True
        return self

    def create_schema(self) -> str:
        return "combined-schema"

    def extract_entities(
        self,
        text: str,
        entity_types: object,
        **options: Any,
    ) -> dict[str, Any]:
        self.calls.append(("extract_entities", (text, entity_types), options))
        return {"entities": {"person": []}}

    def extract_entities_long(
        self,
        text: str,
        entity_types: object,
        **options: Any,
    ) -> dict[str, Any]:
        self.calls.append(("extract_entities_long", (text, entity_types), options))
        return {"entities": {"person": []}}

    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: object,
        **options: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append(("batch_extract_entities", (texts, entity_types), options))
        return [{"entities": {}} for _ in texts]

    def extract(self, text: str, schema: object, **options: Any) -> dict[str, Any]:
        self.calls.append(("extract", (text, schema), options))
        return {"entities": {}}


class _FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def classify(self, text: str, schema: object, *, config: object) -> str:
        self.calls.append((text, schema, config))
        return "classification-result"


class _FakeJointIE:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object, object]] = []

    def create_schema(self) -> str:
        return "joint-schema"

    def extract(self, text: str, schema: object, *, config: object) -> str:
        self.calls.append((text, schema, config))
        return "graph-result"


class _AutoExtractor:
    calls: list[tuple[str, str]] = []
    instance = _FakeExtractor()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        map_location: str,
    ) -> _FakeExtractor:
        cls.calls.append((model_id, map_location))
        return cls.instance


class _AttributeGroup:
    def __init__(self, labels: list[str], **options: Any) -> None:
        self.labels = labels
        self.options = options


class _ClassificationSchema:
    pass


class _ClassificationConfig:
    def __init__(self, **options: Any) -> None:
        self.options = options


class _JointIEConfig:
    def __init__(self, **options: Any) -> None:
        self.options = options


class _TaskBackend:
    name = "fake-ie"

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, Any]]] = []

    def _call(
        self,
        operation: str,
        *args: object,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.calls.append((operation, args, kwargs))
        return {"operation": operation}

    def extract_entities(self, *args: object, **kwargs: Any) -> dict[str, Any]:
        return self._call("extract_entities", *args, **kwargs)

    def extract_entities_long(self, *args: object, **kwargs: Any) -> dict[str, Any]:
        return self._call("extract_entities_long", *args, **kwargs)

    def batch_extract_entities(
        self,
        *args: object,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        result = self._call("batch_extract_entities", *args, **kwargs)
        return [result]

    def extract_schema(self, *args: object, **kwargs: Any) -> dict[str, Any]:
        return self._call("extract_schema", *args, **kwargs)

    def classify_schema(self, *args: object, **kwargs: Any) -> dict[str, Any]:
        return self._call("classify_schema", *args, **kwargs)

    def extract_graph(self, *args: object, **kwargs: Any) -> dict[str, Any]:
        return self._call("extract_graph", *args, **kwargs)


@pytest.fixture(autouse=True)
def _reset_gliner25_cache() -> None:
    clear_gliner25_model_cache()
    _AutoExtractor.calls.clear()
    _AutoExtractor.instance = _FakeExtractor()


def _install_fake_native_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    gliner2 = ModuleType("gliner2")
    gliner2.AutoExtractor = _AutoExtractor  # type: ignore[attr-defined]
    gliner2.AttributeGroup = _AttributeGroup  # type: ignore[attr-defined]

    classification = ModuleType("gliner2.classification")
    classification.ClassificationSchema = _ClassificationSchema  # type: ignore[attr-defined]
    classification.ClassificationConfig = _ClassificationConfig  # type: ignore[attr-defined]
    constraints = ModuleType("gliner2.classification.constraints")
    constraints.implies = lambda left, right: ("implies", left, right)  # type: ignore[attr-defined]
    classification.constraints = constraints  # type: ignore[attr-defined]

    joint_ie = ModuleType("gliner2.joint_ie")
    joint_ie.JointIEConfig = _JointIEConfig  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "gliner2", gliner2)
    monkeypatch.setitem(sys.modules, "gliner2.classification", classification)
    monkeypatch.setitem(sys.modules, "gliner2.classification.constraints", constraints)
    monkeypatch.setitem(sys.modules, "gliner2.joint_ie", joint_ie)


def test_information_extraction_backend_is_discoverable() -> None:
    backend = get_information_extraction_backend("gliner2.5")

    assert backend.name == "gliner25"
    assert backend.default_model == "base"
    assert list_information_extraction_backends() == ["gliner25"]


def test_model_loading_normalizes_device_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_native_modules(monkeypatch)

    first = load_gliner25_model("small", device="gpu")
    second = load_gliner25_model("small", device="cuda")

    assert first is second is _AutoExtractor.instance
    assert first.evaluated is True
    assert _AutoExtractor.calls == [(GLINER25_MODEL_IDS["small"], "cuda")]


def test_backend_delegates_entity_schema_and_batch_calls() -> None:
    backend = GLiNER25Backend()
    fake = _FakeExtractor()
    key = (GLINER25_MODEL_IDS["small"], "cpu")
    gliner25_module._MODEL_CACHE[key] = fake

    assert backend.create_schema(model="small", device="cpu") == "combined-schema"
    assert backend.extract_entities(
        "Alice",
        ["person"],
        model="small",
        device="cpu",
        include_spans=True,
    ) == {"entities": {"person": []}}
    backend.extract_entities_long(
        "long text",
        ["person"],
        model="small",
        device="cpu",
        chunk_size=32,
        chunk_overlap=8,
    )
    batch = backend.batch_extract_entities(
        ["one", "two"],
        ["person"],
        model="small",
        device="cpu",
        batch_size=2,
    )
    backend.extract_schema(
        "Alice",
        "schema",
        model="small",
        device="cpu",
        include_confidence=True,
    )

    assert len(batch) == 2
    assert [call[0] for call in fake.calls] == [
        "extract_entities",
        "extract_entities_long",
        "batch_extract_entities",
        "extract",
    ]
    assert fake.calls[0][2]["include_spans"] is True
    assert fake.calls[1][2]["chunk_size"] == 32
    assert fake.calls[2][2]["batch_size"] == 2


def test_backend_preserves_typed_classifier_and_graph_results() -> None:
    backend = GLiNER25Backend()
    key = (GLINER25_MODEL_IDS["small"], "cpu")
    gliner25_module._MODEL_CACHE[key] = _FakeExtractor()
    classifier = _FakeClassifier()
    joint = _FakeJointIE()
    gliner25_module._CLASSIFIER_CACHE[key] = classifier
    gliner25_module._JOINT_IE_CACHE[key] = joint

    assert backend.classify_schema(
        "request",
        "classification-schema",
        model="small",
        device="cpu",
        config="classification-config",
    ) == "classification-result"
    assert backend.create_joint_schema(model="small", device="cpu") == "joint-schema"
    assert backend.extract_graph(
        "Alice works for Acme",
        "joint-schema",
        model="small",
        device="cpu",
        config="joint-config",
    ) == "graph-result"
    assert classifier.calls == [
        ("request", "classification-schema", "classification-config")
    ]
    assert joint.calls == [
        ("Alice works for Acme", "joint-schema", "joint-config")
    ]


def test_backend_schema_factories_hide_native_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_native_modules(monkeypatch)
    backend = GLiNER25Backend()

    attribute = backend.create_attribute_group(["present", "negated"], multi_label=False)
    classification = backend.create_classification_schema()
    classification_config = backend.create_classification_config(decoder="exact")
    joint_config = backend.create_joint_config(optimizer="beam")

    assert attribute.labels == ["present", "negated"]
    assert attribute.options == {"multi_label": False}
    assert isinstance(classification, _ClassificationSchema)
    assert classification_config.options == {"decoder": "exact"}
    assert joint_config.options == {"optimizer": "beam"}
    assert backend.classification_constraints.implies(("a", "b"), ("c", "d")) == (
        "implies",
        ("a", "b"),
        ("c", "d"),
    )


def test_information_extraction_tasks_delegate_to_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _TaskBackend()
    monkeypatch.setattr(
        task_module,
        "get_information_extraction_backend",
        lambda name: backend,
    )

    assert task_module.extract_entities("Alice", ["person"], model="small") == {
        "operation": "extract_entities"
    }
    assert task_module.extract_entities_long("Alice", ["person"]) == {
        "operation": "extract_entities_long"
    }
    assert task_module.batch_extract_entities(["Alice"], ["person"]) == [
        {"operation": "batch_extract_entities"}
    ]
    assert task_module.extract_schema("Alice", "schema") == {
        "operation": "extract_schema"
    }
    assert task_module.classify_schema("Alice", "schema") == {
        "operation": "classify_schema"
    }
    assert task_module.extract_graph("Alice", "schema") == {
        "operation": "extract_graph"
    }
    assert all("backend" not in call[2] for call in backend.calls)
    assert backend.calls[0][2]["model"] == "small"


def test_gliner25_examples_and_colab_use_aibackends_api() -> None:
    paths = [
        *sorted(Path("examples/gliner25").glob("*.py")),
        Path("evals/eval_gliner25.py"),
        Path("benchmarks/benchmark_gliner25_cpu.py"),
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "from gliner2" not in source
        assert "import gliner2" not in source

    notebook = json.loads(
        Path(
            "examples/notebooks/gliner25_information_extraction_colab.ipynb"
        ).read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'aibackends[gliner2]' in code
    assert "get_information_extraction_backend" in code
    assert "from gliner2" not in code
    assert "import gliner2" not in code

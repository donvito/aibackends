from __future__ import annotations

import threading
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, cast

from aibackends.backends.information_extraction._base import (
    BaseInformationExtractionBackend,
    EntityTypes,
)
from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError

GLINER25_MODEL_IDS = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}
DEFAULT_GLINER25_MODEL = "base"

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_CLASSIFIER_CACHE: dict[tuple[str, str], Any] = {}
_JOINT_IE_CACHE: dict[tuple[str, str], Any] = {}
_INFERENCE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def resolve_model_id(model: str | None = None) -> str:
    """Resolve a GLiNER2.5 alias or accept one of the published full model IDs."""
    value = (model or DEFAULT_GLINER25_MODEL).strip()
    if value in GLINER25_MODEL_IDS:
        return GLINER25_MODEL_IDS[value]
    if value in GLINER25_MODEL_IDS.values():
        return value
    choices = ", ".join(GLINER25_MODEL_IDS)
    raise ValueError(f"Unknown GLiNER2.5 model {model!r}. Use one of: {choices}.")


def normalize_device(device: str) -> str:
    """Normalize explicit device aliases and auto-detect a local accelerator."""
    value = device.strip().lower()
    if value == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if value == "gpu":
        return "cuda"
    if value in {"cpu", "cuda", "mps"}:
        return value
    prefix, separator, index = value.partition(":")
    if prefix == "cuda" and separator and index.isdigit():
        return value
    raise ValueError(
        "Unsupported device. Use 'auto', 'cpu', 'gpu', 'cuda', 'cuda:<index>', or 'mps'."
    )


def _cache_key(model: str | None, device: str) -> tuple[str, str]:
    return resolve_model_id(model), normalize_device(device)


def _dependency_error(exc: ImportError) -> RuntimeImportError:
    return RuntimeImportError(
        "Install 'aibackends[gliner2]' to use the GLiNER2.5 information extraction backend."
    )


def load_gliner25_model(
    model: str | None = None,
    *,
    device: str = "auto",
) -> Any:
    """Load one GLiNER2.5 checkpoint once per process and device."""
    key = _cache_key(model, device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from gliner2 import AutoExtractor
        except ImportError as exc:
            raise _dependency_error(exc) from exc
        extractor = AutoExtractor.from_pretrained(key[0], map_location=key[1])
        evaluate = getattr(extractor, "eval", None)
        if callable(evaluate):
            evaluate()
        _MODEL_CACHE[key] = extractor
        _INFERENCE_LOCKS[key] = threading.Lock()
        return extractor


def _load_classifier(model: str | None, device: str) -> Any:
    key = _cache_key(model, device)
    extractor = load_gliner25_model(model, device=device)
    cached = _CLASSIFIER_CACHE.get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _CLASSIFIER_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from gliner2.classification import Classifier
        except ImportError as exc:
            raise _dependency_error(exc) from exc
        classifier = Classifier(extractor, device=key[1]).eval()
        _CLASSIFIER_CACHE[key] = classifier
        return classifier


def _load_joint_ie(model: str | None, device: str) -> Any:
    key = _cache_key(model, device)
    extractor = load_gliner25_model(model, device=device)
    cached = _JOINT_IE_CACHE.get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _JOINT_IE_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from gliner2.joint_ie import JointIE
        except ImportError as exc:
            raise _dependency_error(exc) from exc
        joint = JointIE(extractor, device=key[1]).eval()
        _JOINT_IE_CACHE[key] = joint
        return joint


def _inference_lock(model: str | None, device: str) -> threading.Lock:
    key = _cache_key(model, device)
    with _CACHE_LOCK:
        return _INFERENCE_LOCKS.setdefault(key, threading.Lock())


def clear_gliner25_model_cache() -> None:
    """Drop all cached extractors and decoders. Intended for tests and memory management."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _CLASSIFIER_CACHE.clear()
        _JOINT_IE_CACHE.clear()
        _INFERENCE_LOCKS.clear()


def result_to_dict(result: object) -> dict[str, Any]:
    """Convert dictionary and typed native results to a serializable dictionary."""
    if isinstance(result, dict):
        return cast(dict[str, Any], result)
    for method_name in ("to_dict", "model_dump"):
        method = getattr(result, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                return cast(dict[str, Any], value)
    raise TypeError(f"Unsupported GLiNER2.5 result type: {type(result).__name__}.")


def iter_spans(value: object) -> Iterator[Mapping[str, object]]:
    """Yield every nested result object with text and half-open character offsets."""
    if isinstance(value, Mapping):
        if {"text", "start", "end"}.issubset(value):
            yield value
            return
        for child in value.values():
            yield from iter_spans(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from iter_spans(child)


def assert_source_spans(text: str, result: object) -> int:
    """Assert that every returned span slices back to identical source text."""
    count = 0
    for span in iter_spans(result_to_dict(result)):
        start = span["start"]
        end = span["end"]
        extracted = span["text"]
        if not isinstance(start, int) or not isinstance(end, int):
            raise AssertionError(f"Non-integer span offsets: {span!r}")
        if not isinstance(extracted, str):
            raise AssertionError(f"Non-string span text: {span!r}")
        if start < 0 or end <= start or end > len(text):
            raise AssertionError(f"Out-of-range span: {span!r}")
        actual = text[start:end]
        if actual != extracted:
            raise AssertionError(
                f"Span mismatch at [{start}:{end}]: expected {extracted!r}, found {actual!r}."
            )
        count += 1
    return count


def _require_dict(result: object, operation: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TaskExecutionError(f"GLiNER2.5 returned an invalid {operation} result.")
    return cast(dict[str, Any], result)


class GLiNER25Backend(BaseInformationExtractionBackend):
    """First-class aibackends facade over GLiNER2.5 boundary checkpoints."""

    name = "gliner25"
    aliases = ("gliner2.5", "gliner2-ie")
    model_ids = GLINER25_MODEL_IDS
    default_model = DEFAULT_GLINER25_MODEL

    def resolve_model_id(self, model: str | None = None) -> str:
        return resolve_model_id(model)

    def normalize_device(self, device: str) -> str:
        return normalize_device(device)

    def load(self, *, model: str | None = None, device: str = "auto") -> Any:
        return load_gliner25_model(model, device=device)

    def create_schema(self, *, model: str | None = None, device: str = "auto") -> Any:
        extractor = self.load(model=model, device=device)
        return extractor.create_schema()

    def create_attribute_group(
        self,
        labels: Sequence[str],
        **options: Any,
    ) -> Any:
        try:
            from gliner2 import AttributeGroup
        except ImportError as exc:
            raise _dependency_error(exc) from exc
        return AttributeGroup(list(labels), **options)

    def create_classification_schema(self) -> Any:
        try:
            from gliner2.classification import ClassificationSchema
        except ImportError as exc:
            raise _dependency_error(exc) from exc
        return ClassificationSchema()

    def create_classification_config(self, **options: Any) -> Any:
        try:
            from gliner2.classification import ClassificationConfig
        except ImportError as exc:
            raise _dependency_error(exc) from exc
        return ClassificationConfig(**options)

    @property
    def classification_constraints(self) -> Any:
        try:
            from gliner2.classification import constraints
        except ImportError as exc:
            raise _dependency_error(exc) from exc
        return constraints

    def create_joint_schema(
        self,
        *,
        model: str | None = None,
        device: str = "auto",
    ) -> Any:
        return _load_joint_ie(model, device).create_schema()

    def create_joint_config(self, **options: Any) -> Any:
        try:
            from gliner2.joint_ie import JointIEConfig
        except ImportError as exc:
            raise _dependency_error(exc) from exc
        return JointIEConfig(**options)

    def extract_entities(
        self,
        text: str,
        entity_types: EntityTypes,
        *,
        model: str | None = None,
        device: str = "auto",
        **options: Any,
    ) -> dict[str, Any]:
        extractor = self.load(model=model, device=device)
        with _inference_lock(model, device):
            result = extractor.extract_entities(text, entity_types, **options)
        return _require_dict(result, "entity extraction")

    def extract_entities_long(
        self,
        text: str,
        entity_types: EntityTypes,
        *,
        model: str | None = None,
        device: str = "auto",
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        **options: Any,
    ) -> dict[str, Any]:
        extractor = self.load(model=model, device=device)
        with _inference_lock(model, device):
            result = extractor.extract_entities_long(
                text,
                entity_types,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                **options,
            )
        return _require_dict(result, "long-document entity extraction")

    def batch_extract_entities(
        self,
        texts: Sequence[str],
        entity_types: EntityTypes,
        *,
        model: str | None = None,
        device: str = "auto",
        batch_size: int = 8,
        **options: Any,
    ) -> list[dict[str, Any]]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if not texts:
            return []
        extractor = self.load(model=model, device=device)
        with _inference_lock(model, device):
            results = extractor.batch_extract_entities(
                list(texts),
                entity_types,
                batch_size=batch_size,
                **options,
            )
        if not isinstance(results, list) or len(results) != len(texts):
            raise TaskExecutionError(
                "GLiNER2.5 returned an invalid number of batch entity results."
            )
        if any(not isinstance(result, dict) for result in results):
            raise TaskExecutionError("GLiNER2.5 returned an invalid batch entity result.")
        return cast(list[dict[str, Any]], results)

    def extract_schema(
        self,
        text: str,
        schema: Any,
        *,
        model: str | None = None,
        device: str = "auto",
        **options: Any,
    ) -> dict[str, Any]:
        extractor = self.load(model=model, device=device)
        with _inference_lock(model, device):
            result = extractor.extract(text, schema, **options)
        return _require_dict(result, "schema extraction")

    def classify_schema(
        self,
        text: str,
        schema: Any,
        *,
        model: str | None = None,
        device: str = "auto",
        config: Any = None,
    ) -> Any:
        classifier = _load_classifier(model, device)
        with _inference_lock(model, device):
            return classifier.classify(text, schema, config=config)

    def extract_graph(
        self,
        text: str,
        schema: Any,
        *,
        model: str | None = None,
        device: str = "auto",
        config: Any = None,
    ) -> Any:
        joint = _load_joint_ie(model, device)
        with _inference_lock(model, device):
            return joint.extract(text, schema, config=config)


GLINER25_BACKEND = GLiNER25Backend()

__all__ = [
    "DEFAULT_GLINER25_MODEL",
    "GLINER25_BACKEND",
    "GLINER25_MODEL_IDS",
    "GLiNER25Backend",
    "assert_source_spans",
    "clear_gliner25_model_cache",
    "iter_spans",
    "load_gliner25_model",
    "normalize_device",
    "resolve_model_id",
    "result_to_dict",
]

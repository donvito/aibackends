from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any

from aibackends.backends.extraction._base import (
    AttributesInput,
    BaseExtractionBackend,
    ConstraintsInput,
    LabelsInput,
    RelationsInput,
    TasksInput,
)
from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.schemas.extraction import (
    EntityExtraction,
    ExtractedEntity,
    GraphEntity,
    GraphRelation,
    KnowledgeGraph,
    SpanAttribute,
    TaskClassification,
    TextClassification,
)

MODEL_VARIANTS: dict[str, str] = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}
DEFAULT_MODEL_VARIANT = "base"
DEFAULT_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 8
DEFAULT_CHUNK_SIZE = 384
DEFAULT_CHUNK_OVERLAP = 64
DEFAULT_BEAM_SIZE = 32

_ENTITY_SPAN_KEYS = {"text", "start", "end", "confidence"}
_CONSTRAINT_KINDS = ("implies", "excludes", "iff")

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_CLASSIFIER_CACHE: dict[tuple[str, str], Any] = {}
_JOINT_CACHE: dict[tuple[str, str], Any] = {}
_INFERENCE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_CACHE_LOCK = threading.Lock()

_INSTALL_HINT = "Install 'aibackends[extraction]' to use the GLiNER2.5 backend."


def normalize_device(device: str | None) -> str:
    normalized = (device or "cpu").strip().lower()
    if normalized == "gpu":
        return "cuda"
    if normalized in {"cpu", "cuda", "mps"}:
        return normalized
    prefix, separator, index = normalized.partition(":")
    if prefix == "cuda" and separator and index.isdigit():
        return normalized
    raise ValueError(
        "Unsupported GLiNER2.5 device. Use 'cpu', 'gpu', 'cuda', 'cuda:<index>', or 'mps'."
    )


def resolve_model_id(model: str | None) -> str:
    """Map a variant name (small/base/multi) or Hugging Face repo id to a model id."""
    if model is None:
        return MODEL_VARIANTS[DEFAULT_MODEL_VARIANT]
    normalized = model.strip()
    if normalized in MODEL_VARIANTS:
        return MODEL_VARIANTS[normalized]
    if "/" in normalized:
        return normalized
    raise ValueError(
        f"Unknown GLiNER2.5 model {model!r}. Use one of "
        f"{sorted(MODEL_VARIANTS)} or a full Hugging Face repo id."
    )


def load_gliner25_model(model: str | None = None, device: str = "cpu") -> Any:
    """Load a GLiNER2.5 extractor once per process, model id, and device."""
    model_id = resolve_model_id(model)
    device_name = normalize_device(device)
    cache_key = (model_id, device_name)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from gliner2 import AutoExtractor
        except ImportError as exc:
            raise RuntimeImportError(_INSTALL_HINT) from exc

        loaded = AutoExtractor.from_pretrained(model_id, map_location=device_name)
        evaluate = getattr(loaded, "eval", None)
        if callable(evaluate):
            evaluate()
        _MODEL_CACHE[cache_key] = loaded
        _INFERENCE_LOCKS[cache_key] = threading.Lock()
        return loaded


def clear_model_cache() -> None:
    """Drop cached GLiNER2.5 models. Intended for tests and memory management."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _CLASSIFIER_CACHE.clear()
        _JOINT_CACHE.clear()
        _INFERENCE_LOCKS.clear()


def _cache_key(model: str | None, device: str) -> tuple[str, str]:
    return (resolve_model_id(model), normalize_device(device))


def _inference_lock(cache_key: tuple[str, str]) -> threading.Lock:
    with _CACHE_LOCK:
        return _INFERENCE_LOCKS.setdefault(cache_key, threading.Lock())


def _load_classifier(model: str | None, device: str) -> Any:
    cache_key = _cache_key(model, device)
    cached = _CLASSIFIER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    extractor = load_gliner25_model(model, device)
    with _CACHE_LOCK:
        cached = _CLASSIFIER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from gliner2.classification import Classifier
        except ImportError as exc:
            raise RuntimeImportError(_INSTALL_HINT) from exc
        classifier = Classifier(extractor)
        _CLASSIFIER_CACHE[cache_key] = classifier
        return classifier


def _load_joint(model: str | None, device: str) -> Any:
    cache_key = _cache_key(model, device)
    cached = _JOINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    extractor = load_gliner25_model(model, device)
    with _CACHE_LOCK:
        cached = _JOINT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from gliner2.joint_ie import JointIE
        except ImportError as exc:
            raise RuntimeImportError(_INSTALL_HINT) from exc
        joint = JointIE(extractor)
        _JOINT_CACHE[cache_key] = joint
        return joint


def _validate_options(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    batch_size: int | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if chunk_size is not None and chunk_overlap is not None:
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size.")


def _normalize_labels(labels: LabelsInput) -> list[str] | dict[str, str]:
    if isinstance(labels, Mapping):
        parsed_map = {str(key): str(value) for key, value in labels.items()}
        if not parsed_map:
            raise ValueError("labels must not be empty.")
        return parsed_map
    parsed = [str(label) for label in labels]
    if not parsed:
        raise ValueError("labels must not be empty.")
    return parsed


def _build_entity_schema(
    extractor: Any,
    labels: list[str] | dict[str, str],
    attributes: AttributesInput | None,
) -> Any:
    schema = extractor.create_schema().entities(labels)
    if not attributes:
        return schema
    try:
        from gliner2 import AttributeGroup
    except ImportError as exc:
        raise RuntimeImportError(_INSTALL_HINT) from exc

    groups: dict[str, Any] = {}
    for group_name, spec in attributes.items():
        if "labels" not in spec:
            raise ValueError(f"Attribute group {group_name!r} must define 'labels'.")
        applies_to = spec.get("applies_to")
        groups[group_name] = AttributeGroup(
            labels=[str(label) for label in spec["labels"]],
            multi_label=bool(spec.get("multi_label", False)),
            threshold=float(spec.get("threshold", 0.5)),
            applies_to=list(applies_to) if applies_to is not None else None,
            qualify_labels=bool(spec.get("qualify_labels", True)),
        )
    return schema.entity_attributes(groups)


def _parse_span_attribute(value: Any) -> SpanAttribute | None:
    if isinstance(value, Mapping):
        label = value.get("label")
        if not isinstance(label, str):
            return None
        confidence = value.get("confidence")
        confidences = (
            {label: float(confidence)} if isinstance(confidence, int | float) else {}
        )
        return SpanAttribute(labels=[label], confidences=confidences)
    if isinstance(value, list):
        labels: list[str] = []
        confidences = {}
        for item in value:
            if not isinstance(item, Mapping) or not isinstance(item.get("label"), str):
                return None
            labels.append(item["label"])
            if isinstance(item.get("confidence"), int | float):
                confidences[item["label"]] = float(item["confidence"])
        return SpanAttribute(labels=labels, confidences=confidences)
    return None


def _parse_entity_result(
    text: str,
    raw: Any,
    *,
    model_id: str,
    backend_name: str,
) -> EntityExtraction:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("entities"), Mapping):
        raise TaskExecutionError("GLiNER2.5 returned an invalid entity extraction result.")
    entities: list[ExtractedEntity] = []
    for label, spans in raw["entities"].items():
        if not isinstance(spans, list):
            raise TaskExecutionError(f"GLiNER2.5 returned invalid spans for {label!r}.")
        for span in spans:
            if isinstance(span, str):
                entities.append(ExtractedEntity(label=str(label), text=span))
                continue
            if not isinstance(span, Mapping) or not isinstance(span.get("text"), str):
                raise TaskExecutionError(f"GLiNER2.5 returned an invalid span for {label!r}.")
            attributes: dict[str, SpanAttribute] = {}
            for key, value in span.items():
                if key in _ENTITY_SPAN_KEYS:
                    continue
                parsed = _parse_span_attribute(value)
                if parsed is not None:
                    attributes[str(key)] = parsed
            entities.append(
                ExtractedEntity(
                    label=str(label),
                    text=span["text"],
                    start=span.get("start"),
                    end=span.get("end"),
                    confidence=span.get("confidence"),
                    attributes=attributes,
                )
            )
    entities.sort(key=lambda entity: (entity.start is None, entity.start or 0))
    return EntityExtraction(
        text=text,
        entities=entities,
        backend_used=backend_name,
        model_id=model_id,
    )


def _build_classification_schema(
    tasks: TasksInput,
    constraints: ConstraintsInput | None,
) -> Any:
    try:
        from gliner2.classification import ClassificationSchema
        from gliner2.classification import constraints as constraint_dsl
    except ImportError as exc:
        raise RuntimeImportError(_INSTALL_HINT) from exc

    if not tasks:
        raise ValueError("tasks must not be empty.")
    schema = ClassificationSchema()
    for name, spec in tasks.items():
        if isinstance(spec, Mapping):
            options = dict(spec)
            labels = options.pop("labels", None)
            if not labels:
                raise ValueError(f"Classification task {name!r} must define 'labels'.")
            multi_label = bool(options.pop("multi_label", False))
            allowed = {"min_labels", "max_labels", "threshold", "default", "instruction"}
            unknown = set(options) - allowed
            if unknown:
                raise ValueError(
                    f"Unknown options for classification task {name!r}: {sorted(unknown)}"
                )
            if multi_label:
                schema.multi(name, [str(label) for label in labels], **options)
            else:
                schema.single(name, [str(label) for label in labels], **options)
        else:
            schema.single(name, [str(label) for label in spec])

    for rule in constraints or ():
        kind = rule.get("kind")
        if kind not in _CONSTRAINT_KINDS:
            raise ValueError(
                f"Unknown constraint kind {kind!r}. Use one of {list(_CONSTRAINT_KINDS)}."
            )
        when = _constraint_ref(rule.get("when"), "when")
        then = _constraint_ref(rule.get("then"), "then")
        builder = getattr(constraint_dsl, kind)
        schema.constrain(builder(when, then))
    return schema


def _constraint_ref(value: Any, field: str) -> tuple[str, str]:
    if (
        isinstance(value, Sequence)
        and not isinstance(value, str)
        and len(value) == 2
        and all(isinstance(item, str) for item in value)
    ):
        return (value[0], value[1])
    raise ValueError(f"Constraint {field!r} must be a [task, label] pair.")


def _parse_classification_result(
    text: str,
    raw: Any,
    *,
    constrained: bool,
    model_id: str,
    backend_name: str,
) -> TextClassification:
    tasks_attr = getattr(raw, "tasks", None)
    if not isinstance(tasks_attr, Mapping):
        raise TaskExecutionError("GLiNER2.5 returned an invalid classification result.")
    parsed: dict[str, TaskClassification] = {}
    for name, task_result in tasks_attr.items():
        labels = [str(label) for label in getattr(task_result, "labels", ())]
        probabilities = {
            str(label): float(probability)
            for label, probability in dict(getattr(task_result, "probabilities", {})).items()
        }
        confidence = getattr(task_result, "confidence", None)
        parsed[str(name)] = TaskClassification(
            task=str(name),
            labels=labels,
            multi_label=not bool(getattr(task_result, "exclusive", True)),
            confidence=float(confidence) if confidence is not None else None,
            probabilities=probabilities,
        )
    return TextClassification(
        text=text,
        tasks=parsed,
        feasible=bool(getattr(raw, "feasible", True)),
        constrained=constrained,
        backend_used=backend_name,
        model_id=model_id,
    )


def _validate_relations(relations: RelationsInput) -> None:
    if not relations:
        raise ValueError("relations must not be empty.")
    allowed = {"name", "head", "tail", "description", "threshold", "unique_head", "unique_tail"}
    for relation in relations:
        name = relation.get("name")
        if not isinstance(name, str) or relation.get("head") is None or (
            relation.get("tail") is None
        ):
            raise ValueError("Each relation must define 'name', 'head', and 'tail'.")
        unknown = set(relation) - allowed
        if unknown:
            raise ValueError(f"Unknown options for relation {name!r}: {sorted(unknown)}")


def _build_joint_schema(
    joint: Any,
    entities: list[str] | dict[str, str],
    relations: RelationsInput,
    *,
    no_self_loops: bool,
) -> Any:
    schema = joint.create_schema().entities(entities)
    for relation in relations:
        options = dict(relation)
        name = options.pop("name")
        head = options.pop("head")
        tail = options.pop("tail")
        schema.relation(name, head, tail, **options)
    if no_self_loops:
        schema.no_self_loops()
    return schema


def _parse_graph_result(
    text: str,
    raw: Any,
    *,
    model_id: str,
    backend_name: str,
) -> KnowledgeGraph:
    raw_entities = getattr(raw, "entities", None)
    raw_relations = getattr(raw, "relations", None)
    if raw_entities is None or raw_relations is None:
        raise TaskExecutionError("GLiNER2.5 returned an invalid joint extraction result.")
    entities = [
        GraphEntity(
            id=str(entity.id),
            type=str(entity.type),
            text=str(entity.text),
            start=getattr(entity, "start", None),
            end=getattr(entity, "end", None),
            confidence=getattr(entity, "confidence", None),
        )
        for entity in raw_entities
    ]
    entity_texts = {entity.id: entity.text for entity in entities}
    relations = [
        GraphRelation(
            type=str(relation.type),
            head=str(relation.head),
            tail=str(relation.tail),
            head_text=entity_texts.get(str(relation.head), ""),
            tail_text=entity_texts.get(str(relation.tail), ""),
            confidence=getattr(relation, "confidence", None),
        )
        for relation in raw_relations
    ]
    return KnowledgeGraph(
        text=text,
        entities=entities,
        relations=relations,
        feasible=bool(getattr(raw, "feasible", True)),
        backend_used=backend_name,
        model_id=model_id,
    )


class Gliner25Backend(BaseExtractionBackend):
    name = "gliner2.5"
    aliases = ("gliner25",)
    default_model = MODEL_VARIANTS[DEFAULT_MODEL_VARIANT]

    def resolve_model_id(self, model: str | None) -> str:
        return resolve_model_id(model)

    def load(self, *, model: str | None = None, device: str = "cpu") -> Any:
        return load_gliner25_model(model, device)

    def extract_entities(
        self,
        text: str,
        *,
        labels: LabelsInput,
        attributes: AttributesInput | None = None,
        model: str | None = None,
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
        long_document: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> EntityExtraction:
        results = self.extract_entities_batch(
            [text],
            labels=labels,
            attributes=attributes,
            model=model,
            device=device,
            threshold=threshold,
            long_document=long_document,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return results[0]

    def extract_entities_batch(
        self,
        texts: Sequence[str],
        *,
        labels: LabelsInput,
        attributes: AttributesInput | None = None,
        model: str | None = None,
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
        long_document: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> list[EntityExtraction]:
        _validate_options(
            threshold=threshold,
            batch_size=batch_size,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        normalized_labels = _normalize_labels(labels)
        if not texts:
            return []
        cache_key = _cache_key(model, device)
        extractor = load_gliner25_model(model, device)
        schema = _build_entity_schema(extractor, normalized_labels, attributes)
        with _inference_lock(cache_key):
            if long_document:
                raw_results = extractor.batch_extract_long(
                    list(texts),
                    schema,
                    batch_size=batch_size,
                    threshold=threshold,
                    include_confidence=True,
                    include_spans=True,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:
                raw_results = extractor.batch_extract(
                    list(texts),
                    schema,
                    batch_size=batch_size,
                    threshold=threshold,
                    include_confidence=True,
                    include_spans=True,
                )
        if not isinstance(raw_results, list) or len(raw_results) != len(texts):
            raise TaskExecutionError("GLiNER2.5 returned an invalid number of batch results.")
        model_id = cache_key[0]
        return [
            _parse_entity_result(text, raw, model_id=model_id, backend_name=self.name)
            for text, raw in zip(texts, raw_results, strict=True)
        ]

    def classify_text(
        self,
        text: str,
        *,
        tasks: TasksInput,
        constraints: ConstraintsInput | None = None,
        model: str | None = None,
        device: str = "cpu",
    ) -> TextClassification:
        results = self.classify_text_batch(
            [text],
            tasks=tasks,
            constraints=constraints,
            model=model,
            device=device,
        )
        return results[0]

    def classify_text_batch(
        self,
        texts: Sequence[str],
        *,
        tasks: TasksInput,
        constraints: ConstraintsInput | None = None,
        model: str | None = None,
        device: str = "cpu",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[TextClassification]:
        _validate_options(batch_size=batch_size)
        if not texts:
            return []
        cache_key = _cache_key(model, device)
        schema = _build_classification_schema(tasks, constraints)
        classifier = _load_classifier(model, device)
        config = self._classification_config(batch_size)
        with _inference_lock(cache_key):
            raw_results = classifier.batch_classify(list(texts), schema, config=config)
        if not isinstance(raw_results, list) or len(raw_results) != len(texts):
            raise TaskExecutionError("GLiNER2.5 returned an invalid number of batch results.")
        constrained = bool(constraints)
        model_id = cache_key[0]
        return [
            _parse_classification_result(
                text,
                raw,
                constrained=constrained,
                model_id=model_id,
                backend_name=self.name,
            )
            for text, raw in zip(texts, raw_results, strict=True)
        ]

    def extract_graph(
        self,
        text: str,
        *,
        entities: LabelsInput,
        relations: RelationsInput,
        no_self_loops: bool = True,
        model: str | None = None,
        device: str = "cpu",
        optimizer: str = "beam",
        beam_size: int = DEFAULT_BEAM_SIZE,
    ) -> KnowledgeGraph:
        if beam_size < 1:
            raise ValueError("beam_size must be at least 1.")
        normalized_entities = _normalize_labels(entities)
        _validate_relations(relations)
        cache_key = _cache_key(model, device)
        joint = _load_joint(model, device)
        schema = _build_joint_schema(
            joint,
            normalized_entities,
            relations,
            no_self_loops=no_self_loops,
        )
        config = self._joint_config(optimizer, beam_size)
        with _inference_lock(cache_key):
            raw = joint.extract(text, schema, config=config)
        return _parse_graph_result(
            text,
            raw,
            model_id=cache_key[0],
            backend_name=self.name,
        )

    @staticmethod
    def _classification_config(batch_size: int) -> Any:
        try:
            from gliner2.classification import ClassificationConfig
        except ImportError as exc:
            raise RuntimeImportError(_INSTALL_HINT) from exc
        return ClassificationConfig(batch_size=batch_size)

    @staticmethod
    def _joint_config(optimizer: str, beam_size: int) -> Any:
        try:
            from gliner2.joint_ie import JointIEConfig
        except ImportError as exc:
            raise RuntimeImportError(_INSTALL_HINT) from exc
        return JointIEConfig(optimizer=optimizer, beam_size=beam_size)


GLINER25_BACKEND = Gliner25Backend()


__all__ = [
    "DEFAULT_MODEL_VARIANT",
    "GLINER25_BACKEND",
    "Gliner25Backend",
    "MODEL_VARIANTS",
    "clear_model_cache",
    "load_gliner25_model",
    "normalize_device",
    "resolve_model_id",
]

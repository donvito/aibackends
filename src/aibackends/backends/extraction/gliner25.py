from __future__ import annotations

import threading
from typing import Any, Literal, cast

from aibackends.backends.extraction._base import BaseExtractionBackend
from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.schemas.extraction import (
    AttributeSpec,
    ClassificationConstraint,
    ClassificationTaskResult,
    ClassificationTaskSpec,
    ConstrainedClassification,
    EntityExtraction,
    ExtractedEntity,
    ExtractedRelation,
    GraphEntity,
    GraphRelation,
    KnowledgeGraph,
    RelationExtraction,
    RelationSpec,
    SpanAttribute,
)
from aibackends.schemas.pii import PIIEntity

GLINER25_MODELS: dict[str, str] = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}
DEFAULT_MODEL_ALIAS = "base"
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHUNK_SIZE = 384
DEFAULT_CHUNK_OVERLAP = 64
LONG_DOC_WORD_THRESHOLD = 384
INSTALL_HINT = "Install 'aibackends[extraction]' to use the GLiNER2.5 backend."

Kind = Literal["extractor", "classifier", "joint"]

_MODEL_CACHE: dict[tuple[Kind, str, str], Any] = {}
_INFERENCE_LOCKS: dict[tuple[Kind, str, str], threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


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
    if model is None or not model.strip():
        return GLINER25_MODELS[DEFAULT_MODEL_ALIAS]
    key = model.strip()
    alias = key.lower()
    if alias in GLINER25_MODELS:
        return GLINER25_MODELS[alias]
    return key


def word_count(text: str) -> int:
    return len(text.split())


def should_use_long_document(text: str, long_document: bool | None) -> bool:
    if long_document is not None:
        return long_document
    return word_count(text) > LONG_DOC_WORD_THRESHOLD


def clear_model_cache() -> None:
    """Drop cached GLiNER2.5 models. Intended for tests and memory management."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _INFERENCE_LOCKS.clear()


def _cache_key(kind: Kind, model_id: str, device_name: str) -> tuple[Kind, str, str]:
    return (kind, model_id, device_name)


def _inference_lock(kind: Kind, model_id: str, device_name: str) -> threading.Lock:
    key = _cache_key(kind, model_id, device_name)
    with _CACHE_LOCK:
        return _INFERENCE_LOCKS.setdefault(key, threading.Lock())


def _import_gliner2() -> Any:
    try:
        import gliner2
    except ImportError as exc:
        raise RuntimeImportError(INSTALL_HINT) from exc
    return gliner2


def _eval_model(model: Any) -> Any:
    evaluate = getattr(model, "eval", None)
    if callable(evaluate):
        evaluate()
    return model


def _load_cached(
    kind: Kind,
    model_id: str,
    device_name: str,
    factory: Any,
) -> Any:
    key = _cache_key(kind, model_id, device_name)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        model = _eval_model(factory(model_id, device_name))
        _MODEL_CACHE[key] = model
        _INFERENCE_LOCKS[key] = threading.Lock()
        return model


def _from_pretrained(factory: Any, model_id: str, device_name: str) -> Any:
    try:
        return factory.from_pretrained(model_id, map_location=device_name)
    except TypeError:
        model = factory.from_pretrained(model_id)
        mover = getattr(model, "to", None)
        if callable(mover):
            mover(device_name)
        return model


def load_extractor(device: str = "cpu", model: str | None = None) -> Any:
    """Load a GLiNER2.5 AutoExtractor once per process, model, and device."""
    device_name = normalize_device(device)
    model_id = resolve_model_id(model)
    gliner2 = _import_gliner2()

    def _factory(resolved_id: str, map_location: str) -> Any:
        return _from_pretrained(gliner2.AutoExtractor, resolved_id, map_location)

    return _load_cached("extractor", model_id, device_name, _factory)


def load_classifier(device: str = "cpu", model: str | None = None) -> Any:
    """Load a GLiNER2.5 constrained Classifier once per process and device."""
    device_name = normalize_device(device)
    model_id = resolve_model_id(model)

    def _factory(resolved_id: str, map_location: str) -> Any:
        try:
            from gliner2.classification import Classifier
        except ImportError as exc:
            raise RuntimeImportError(INSTALL_HINT) from exc
        return _from_pretrained(Classifier, resolved_id, map_location)

    return _load_cached("classifier", model_id, device_name, _factory)


def load_joint(device: str = "cpu", model: str | None = None) -> Any:
    """Load a GLiNER2.5 JointIE decoder once per process and device."""
    device_name = normalize_device(device)
    model_id = resolve_model_id(model)

    def _factory(resolved_id: str, map_location: str) -> Any:
        try:
            from gliner2.joint_ie import JointIE
        except ImportError as exc:
            raise RuntimeImportError(INSTALL_HINT) from exc
        return _from_pretrained(JointIE, resolved_id, map_location)

    return _load_cached("joint", model_id, device_name, _factory)


def _validate_options(
    *,
    threshold: float | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    beam_size: int | None = None,
) -> None:
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if chunk_size is not None and chunk_size < 1:
        raise ValueError("chunk_size must be at least 1.")
    if chunk_overlap is not None and chunk_overlap < 0:
        raise ValueError("chunk_overlap must be at least 0.")
    if (
        chunk_size is not None
        and chunk_overlap is not None
        and chunk_overlap >= chunk_size
    ):
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    if beam_size is not None and beam_size < 1:
        raise ValueError("beam_size must be at least 1.")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def _span_attributes(item: dict[str, Any]) -> dict[str, SpanAttribute]:
    reserved = {"text", "start", "end", "confidence", "label", "type", "id"}
    attributes: dict[str, SpanAttribute] = {}
    for key, raw in item.items():
        if key in reserved:
            continue
        if isinstance(raw, dict) and "label" in raw:
            attributes[key] = SpanAttribute(
                label=str(raw["label"]),
                confidence=_as_float(raw.get("confidence")),
            )
        elif isinstance(raw, str):
            attributes[key] = SpanAttribute(label=raw)
    return attributes


def _entity_from_span(entity_type: str, item: Any, text: str) -> ExtractedEntity | None:
    if isinstance(item, str):
        start = text.find(item)
        if start < 0:
            return None
        return ExtractedEntity(
            entity_type=entity_type,
            text=item,
            start=start,
            end=start + len(item),
        )
    if not isinstance(item, dict):
        return None
    span_text = str(item.get("text", ""))
    start = _as_int(item.get("start"), -1)
    end = _as_int(item.get("end"), -1)
    slice_matches = 0 <= start < end <= len(text) and (
        not span_text or text[start:end] == span_text
    )
    if slice_matches:
        resolved = text[start:end]
    elif span_text:
        start = text.find(span_text)
        end = start + len(span_text) if start >= 0 else -1
        resolved = span_text
    else:
        return None
    if start < 0 or end <= start or end > len(text):
        return None
    return ExtractedEntity(
        entity_type=str(item.get("type") or item.get("label") or entity_type),
        text=resolved,
        start=start,
        end=end,
        confidence=_as_float(item.get("confidence")),
        attributes=_span_attributes(item),
    )


def parse_entity_result(raw: Any, text: str) -> list[ExtractedEntity]:
    if not isinstance(raw, dict):
        raise TaskExecutionError("GLiNER2.5 returned an invalid entity result.")
    grouped = raw.get("entities", raw)
    if not isinstance(grouped, dict):
        raise TaskExecutionError("GLiNER2.5 returned an invalid entity mapping.")
    entities: list[ExtractedEntity] = []
    for entity_type, items in grouped.items():
        if not isinstance(items, list):
            continue
        for item in items:
            entity = _entity_from_span(str(entity_type), item, text)
            if entity is not None:
                entities.append(entity)
    entities.sort(key=lambda item: (item.start, item.end, item.entity_type))
    return entities


def parse_relation_result(raw: Any, text: str) -> list[ExtractedRelation]:
    if not isinstance(raw, dict):
        raise TaskExecutionError("GLiNER2.5 returned an invalid relation result.")
    grouped = raw.get("relation_extraction", raw)
    if not isinstance(grouped, dict):
        raise TaskExecutionError("GLiNER2.5 returned an invalid relation mapping.")
    relations: list[ExtractedRelation] = []
    for relation_type, items in grouped.items():
        if relation_type in {"entities", "_meta"} or not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            head = _entity_from_span("entity", item.get("head"), text)
            tail = _entity_from_span("entity", item.get("tail"), text)
            if head is None or tail is None:
                continue
            relations.append(
                ExtractedRelation(
                    relation_type=str(relation_type),
                    head=head,
                    tail=tail,
                    confidence=_as_float(item.get("confidence")),
                )
            )
    return relations


def parse_graph_result(raw: Any, text: str) -> tuple[list[GraphEntity], list[GraphRelation], bool]:
    payload: dict[str, Any]
    if hasattr(raw, "to_dict") and callable(raw.to_dict):
        converted = raw.to_dict()
        if not isinstance(converted, dict):
            raise TaskExecutionError("GLiNER2.5 returned an invalid joint IE result.")
        payload = converted
        feasible = bool(getattr(raw, "feasible", payload.get("feasible", True)))
    elif isinstance(raw, dict):
        payload = raw
        feasible = bool(payload.get("feasible", True))
    else:
        raise TaskExecutionError("GLiNER2.5 returned an invalid joint IE result.")

    entities: list[GraphEntity] = []
    for index, item in enumerate(payload.get("entities", []), start=1):
        if not isinstance(item, dict):
            continue
        span_text = str(item.get("text", ""))
        start = _as_int(item.get("start"), -1)
        end = _as_int(item.get("end"), -1)
        slice_matches = 0 <= start < end <= len(text) and (
            not span_text or text[start:end] == span_text
        )
        if slice_matches:
            resolved = text[start:end]
        elif span_text:
            start = text.find(span_text)
            end = start + len(span_text) if start >= 0 else -1
            resolved = span_text
        else:
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        entities.append(
            GraphEntity(
                id=str(item.get("id") or f"e{index}"),
                entity_type=str(item.get("type") or item.get("label") or "entity"),
                text=resolved,
                start=start,
                end=end,
                confidence=_as_float(item.get("confidence")),
            )
        )

    relations: list[GraphRelation] = []
    known_ids = {entity.id for entity in entities}
    for item in payload.get("relations", []):
        if not isinstance(item, dict):
            continue
        head = str(item.get("head", ""))
        tail = str(item.get("tail", ""))
        if head not in known_ids or tail not in known_ids:
            continue
        relations.append(
            GraphRelation(
                relation_type=str(item.get("type") or item.get("label") or "related_to"),
                head=head,
                tail=tail,
                confidence=_as_float(item.get("confidence")),
            )
        )
    return entities, relations, feasible


def parse_classification_result(raw: Any) -> tuple[list[ClassificationTaskResult], bool]:
    payload: dict[str, Any]
    feasible = True
    if hasattr(raw, "to_dict") and callable(raw.to_dict):
        converted = raw.to_dict()
        if not isinstance(converted, dict):
            raise TaskExecutionError("GLiNER2.5 returned an invalid classification result.")
        payload = converted
        feasible = bool(getattr(raw, "feasible", True))
    elif isinstance(raw, dict):
        payload = raw
    else:
        raise TaskExecutionError("GLiNER2.5 returned an invalid classification result.")

    meta = payload.get("_meta")
    if isinstance(meta, dict) and "feasible" in meta:
        feasible = bool(meta["feasible"])

    tasks: list[ClassificationTaskResult] = []
    for name, item in payload.items():
        if name == "_meta":
            continue
        if isinstance(item, dict) and "value" in item:
            value = item["value"]
            if not isinstance(value, (str, list)):
                continue
            probabilities = item.get("probabilities")
            tasks.append(
                ClassificationTaskResult(
                    task=str(name),
                    value=cast(str | list[str], value),
                    confidence=_as_float(item.get("confidence")),
                    probabilities=(
                        {str(key): float(score) for key, score in probabilities.items()}
                        if isinstance(probabilities, dict)
                        else {}
                    ),
                )
            )
        elif isinstance(item, str):
            tasks.append(ClassificationTaskResult(task=str(name), value=item))
        elif isinstance(item, list) and all(isinstance(label, str) for label in item):
            tasks.append(
                ClassificationTaskResult(task=str(name), value=cast(list[str], item))
            )
    return tasks, feasible


def detect_pii_entities(
    text: str,
    labels: list[str],
    *,
    model: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[PIIEntity]:
    """Run GLiNER2.5 entity extraction and map spans to PII entities."""
    backend = GLiNER25Backend()
    extraction = backend.extract_entities(
        text,
        labels,
        model=model,
        threshold=threshold,
    )
    entities: list[PIIEntity] = []
    for item in extraction.entities:
        entities.append(
            PIIEntity(
                entity_type=item.entity_type.upper().replace(" ", "_"),
                text=item.text,
                start=item.start,
                end=item.end,
                replacement="",
            )
        )
    return entities


def _build_classification_schema(
    tasks: list[ClassificationTaskSpec],
    constraints: list[ClassificationConstraint],
) -> Any:
    try:
        from gliner2.classification import ClassificationSchema
        from gliner2.classification import constraints as constraint_lib
    except ImportError as exc:
        raise RuntimeImportError(INSTALL_HINT) from exc

    schema = ClassificationSchema()
    for task in tasks:
        if task.multi_label:
            kwargs: dict[str, Any] = {}
            if task.min_labels is not None:
                kwargs["min_labels"] = task.min_labels
            if task.max_labels is not None:
                kwargs["max_labels"] = task.max_labels
            schema = schema.multi(task.name, list(task.labels), **kwargs)
        else:
            schema = schema.single(task.name, list(task.labels))
    if constraints:
        built = []
        for item in constraints:
            if item.kind == "implies":
                built.append(constraint_lib.implies(item.source, item.target))
            else:
                built.append(constraint_lib.excludes(item.source, item.target))
        schema = schema.constrain(*built)
    return schema


def _build_joint_schema(
    joint: Any,
    entities: list[str],
    relations: list[RelationSpec],
    *,
    no_self_loops: bool,
) -> Any:
    schema = joint.create_schema().entities(list(entities))
    for relation in relations:
        schema = schema.relation(
            relation.name,
            relation.head_type,
            relation.tail_type,
            unique_head=relation.unique_head,
            unique_tail=relation.unique_tail,
        )
    if no_self_loops:
        schema = schema.no_self_loops()
    return schema


def _build_attribute_schema(
    extractor: Any,
    labels: list[str] | dict[str, str],
    attributes: list[AttributeSpec],
) -> Any:
    gliner2 = _import_gliner2()
    attribute_group = getattr(gliner2, "AttributeGroup", None)
    if attribute_group is None:
        raise TaskExecutionError("This gliner2 version does not export AttributeGroup.")
    groups: dict[str, Any] = {}
    for spec in attributes:
        kwargs: dict[str, Any] = {"qualify_labels": spec.qualify_labels}
        if spec.applies_to is not None:
            kwargs["applies_to"] = list(spec.applies_to)
        groups[spec.name] = attribute_group(list(spec.labels), **kwargs)
    return (
        extractor.create_schema()
        .entities(labels)
        .entity_attributes(groups)
    )


class GLiNER25Backend(BaseExtractionBackend):
    name = "gliner25"
    aliases = ("gliner2.5", "gliner-2.5", "gliner2_5")

    def resolve_model_id(self, model: str | None) -> str:
        return resolve_model_id(model)

    def load(self, *, device: str = "cpu", model: str | None = None) -> Any:
        return load_extractor(device=device, model=model)

    def extract_entities(
        self,
        text: str,
        labels: list[str] | dict[str, str],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        long_document: bool | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> EntityExtraction:
        _validate_options(
            threshold=threshold,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        device_name = normalize_device(device)
        model_id = resolve_model_id(model)
        extractor = load_extractor(device=device_name, model=model_id)
        use_long = should_use_long_document(text, long_document)
        kwargs: dict[str, Any] = {
            "include_confidence": True,
            "include_spans": True,
            "threshold": threshold,
        }
        method_name = "extract_entities_long" if use_long else "extract_entities"
        method = getattr(extractor, method_name, None)
        if not callable(method):
            raise TaskExecutionError(f"GLiNER2.5 extractor is missing {method_name}().")
        if use_long:
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
        with _inference_lock("extractor", model_id, device_name):
            raw = method(text, labels, **kwargs)
        return EntityExtraction(
            text=text,
            entities=parse_entity_result(raw, text),
            backend_used=self.name,
            model_id=model_id,
            long_document=use_long,
        )

    def extract_relations(
        self,
        text: str,
        relations: list[str],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> RelationExtraction:
        _validate_options(threshold=threshold)
        device_name = normalize_device(device)
        model_id = resolve_model_id(model)
        extractor = load_extractor(device=device_name, model=model_id)
        method = getattr(extractor, "extract_relations", None)
        if not callable(method):
            raise TaskExecutionError("GLiNER2.5 extractor is missing extract_relations().")
        with _inference_lock("extractor", model_id, device_name):
            raw = method(
                text,
                list(relations),
                include_spans=True,
                include_confidence=True,
                threshold=threshold,
            )
        return RelationExtraction(
            text=text,
            relations=parse_relation_result(raw, text),
            backend_used=self.name,
            model_id=model_id,
        )

    def extract_graph(
        self,
        text: str,
        entities: list[str],
        relations: list[RelationSpec],
        *,
        device: str = "cpu",
        model: str | None = None,
        no_self_loops: bool = True,
        long_document: bool | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        beam_size: int = 32,
    ) -> KnowledgeGraph:
        _validate_options(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            beam_size=beam_size,
        )
        device_name = normalize_device(device)
        model_id = resolve_model_id(model)
        joint = load_joint(device=device_name, model=model_id)
        schema = _build_joint_schema(
            joint,
            entities,
            relations,
            no_self_loops=no_self_loops,
        )
        use_long = should_use_long_document(text, long_document)
        try:
            from gliner2.joint_ie import JointIEConfig
        except ImportError as exc:
            raise RuntimeImportError(INSTALL_HINT) from exc
        config = JointIEConfig(optimizer="beam", beam_size=beam_size)
        method_name = "extract_long" if use_long else "extract"
        method = getattr(joint, method_name, None)
        if not callable(method):
            raise TaskExecutionError(f"GLiNER2.5 JointIE is missing {method_name}().")
        kwargs: dict[str, Any] = {"config": config}
        if use_long:
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
        with _inference_lock("joint", model_id, device_name):
            raw = method(text, schema, **kwargs)
        graph_entities, graph_relations, feasible = parse_graph_result(raw, text)
        return KnowledgeGraph(
            text=text,
            entities=graph_entities,
            relations=graph_relations,
            feasible=feasible,
            backend_used=self.name,
            model_id=model_id,
            long_document=use_long,
        )

    def classify_constrained(
        self,
        text: str,
        tasks: list[ClassificationTaskSpec],
        constraints: list[ClassificationConstraint],
        *,
        device: str = "cpu",
        model: str | None = None,
        long_document: bool | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        decoder: str = "exact",
        beam_size: int = 16,
    ) -> ConstrainedClassification:
        _validate_options(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            beam_size=beam_size,
        )
        if not tasks:
            raise ValueError("classify_constrained requires at least one task.")
        device_name = normalize_device(device)
        model_id = resolve_model_id(model)
        classifier = load_classifier(device=device_name, model=model_id)
        schema = _build_classification_schema(tasks, constraints)
        try:
            from gliner2.classification import ClassificationConfig
        except ImportError as exc:
            raise RuntimeImportError(INSTALL_HINT) from exc
        config = ClassificationConfig(decoder=decoder, beam_size=beam_size)
        use_long = should_use_long_document(text, long_document)
        method_name = "classify_long" if use_long else "classify"
        method = getattr(classifier, method_name, None)
        if not callable(method):
            raise TaskExecutionError(f"GLiNER2.5 Classifier is missing {method_name}().")
        kwargs: dict[str, Any] = {"config": config}
        if use_long:
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
        with _inference_lock("classifier", model_id, device_name):
            raw = method(text, schema, **kwargs)
        results, feasible = parse_classification_result(raw)
        return ConstrainedClassification(
            text=text,
            tasks=results,
            feasible=feasible,
            backend_used=self.name,
            model_id=model_id,
        )

    def extract_with_attributes(
        self,
        text: str,
        labels: list[str] | dict[str, str],
        attributes: list[AttributeSpec],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        long_document: bool | None = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> EntityExtraction:
        _validate_options(
            threshold=threshold,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not attributes:
            return self.extract_entities(
                text,
                labels,
                device=device,
                model=model,
                threshold=threshold,
                long_document=long_document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        device_name = normalize_device(device)
        model_id = resolve_model_id(model)
        extractor = load_extractor(device=device_name, model=model_id)
        schema = _build_attribute_schema(extractor, labels, attributes)
        use_long = should_use_long_document(text, long_document)
        method_name = "extract_long" if use_long else "extract"
        method = getattr(extractor, method_name, None)
        if not callable(method):
            raise TaskExecutionError(f"GLiNER2.5 extractor is missing {method_name}().")
        kwargs: dict[str, Any] = {
            "include_spans": True,
            "include_confidence": True,
            "threshold": threshold,
        }
        if use_long:
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
        with _inference_lock("extractor", model_id, device_name):
            raw = method(text, schema, **kwargs)
        return EntityExtraction(
            text=text,
            entities=parse_entity_result(raw, text),
            backend_used=self.name,
            model_id=model_id,
            long_document=use_long,
        )


GLINER25_BACKEND = GLiNER25Backend()

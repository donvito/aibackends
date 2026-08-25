from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from typing import Any, cast

from aibackends.backends.extraction._base import BaseExtractionBackend
from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.schemas.extraction import (
    EntityExtraction,
    ExtractedEntity,
    GraphEntity,
    GraphExtraction,
    GraphRelation,
    RecordExtraction,
    SchemaClassification,
    SpanAttribute,
    TaskClassification,
)
from aibackends.schemas.pii import PIIEntity

DEFAULT_GLINER25_MODEL = "fastino/gliner2.5-small-v1"
DEFAULT_THRESHOLD = 0.5
DEFAULT_CHUNK_SIZE = 384
DEFAULT_CHUNK_OVERLAP = 64
BACKEND_NAME = "gliner25"

GLINER25_ALIASES: dict[str, str] = {
    "gliner25-small": "fastino/gliner2.5-small-v1",
    "gliner25-base": "fastino/gliner2.5-base-v1",
    "gliner25-multi": "fastino/gliner2.5-multi-v1",
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_CLASSIFIER_CACHE: dict[tuple[str, str], Any] = {}
_JOINT_CACHE: dict[tuple[str, str], Any] = {}
_INFERENCE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def resolve_gliner25_model_id(model: str | None) -> str:
    if model is None or not str(model).strip():
        return DEFAULT_GLINER25_MODEL
    key = str(model).strip()
    return GLINER25_ALIASES.get(key, key)


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
        "Unsupported GLiNER 2.5 device. Use 'cpu', 'gpu', 'cuda', "
        "'cuda:<index>', or 'mps'."
    )


def _import_auto_extractor() -> Any:
    try:
        from gliner2 import AutoExtractor
    except ImportError as exc:
        raise RuntimeImportError(
            "Install 'aibackends[gliner25]' to use the GLiNER 2.5 backend."
        ) from exc
    return AutoExtractor


def load_gliner25_extractor(model_id: str, device: str = "cpu") -> Any:
    """Load a GLiNER 2.5 extractor once per process, model, and device."""
    device_name = normalize_device(device)
    resolved = resolve_gliner25_model_id(model_id)
    cache_key = (resolved, device_name)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        auto_extractor = _import_auto_extractor()
        model = auto_extractor.from_pretrained(
            resolved,
            map_location=device_name,
        )
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        _MODEL_CACHE[cache_key] = model
        _INFERENCE_LOCKS[cache_key] = threading.Lock()
        return model


def load_gliner25_classifier(model_id: str, device: str = "cpu") -> Any:
    extractor = load_gliner25_extractor(model_id, device)
    device_name = normalize_device(device)
    resolved = resolve_gliner25_model_id(model_id)
    cache_key = (resolved, device_name)
    cached = _CLASSIFIER_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _CLASSIFIER_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from gliner2.classification import Classifier
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[gliner25]' for constrained classification."
            ) from exc
        classifier = Classifier(extractor, device=device_name)
        _CLASSIFIER_CACHE[cache_key] = classifier
        return classifier


def load_gliner25_joint(model_id: str, device: str = "cpu") -> Any:
    extractor = load_gliner25_extractor(model_id, device)
    device_name = normalize_device(device)
    resolved = resolve_gliner25_model_id(model_id)
    cache_key = (resolved, device_name)
    cached = _JOINT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    with _CACHE_LOCK:
        cached = _JOINT_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from gliner2.joint_ie import JointIE
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[gliner25]' for joint information extraction."
            ) from exc
        joint = JointIE(extractor, device=device_name)
        _JOINT_CACHE[cache_key] = joint
        return joint


def clear_model_cache() -> None:
    """Drop cached GLiNER 2.5 models. Intended for tests and memory management."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _CLASSIFIER_CACHE.clear()
        _JOINT_CACHE.clear()
        _INFERENCE_LOCKS.clear()


def _inference_lock(model_id: str, device: str) -> threading.Lock:
    resolved = resolve_gliner25_model_id(model_id)
    device_name = normalize_device(device)
    cache_key = (resolved, device_name)
    with _CACHE_LOCK:
        return _INFERENCE_LOCKS.setdefault(cache_key, threading.Lock())


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _span_item(item: Any) -> tuple[str, int | None, int | None, float | None, dict[str, Any]]:
    if isinstance(item, str):
        return item, None, None, None, {}
    if not isinstance(item, dict):
        raise TaskExecutionError(f"GLiNER 2.5 returned an invalid span item: {item!r}.")
    text = str(item.get("text", ""))
    start = item.get("start")
    end = item.get("end")
    confidence = item.get("confidence")
    extras = {
        key: value
        for key, value in item.items()
        if key not in {"text", "start", "end", "confidence", "label"}
    }
    start_i = int(start) if isinstance(start, (int, float)) else None
    end_i = int(end) if isinstance(end, (int, float)) else None
    conf_f = float(confidence) if isinstance(confidence, (int, float)) else None
    return text, start_i, end_i, conf_f, extras


def _parse_attribute(name: str, value: Any) -> SpanAttribute:
    if isinstance(value, str):
        return SpanAttribute(name=name, label=value)
    if isinstance(value, list):
        labels = [str(item) for item in value]
        return SpanAttribute(name=name, label=labels)
    if isinstance(value, dict):
        label = value.get("label", value.get("value"))
        confidence = value.get("confidence")
        conf_f = float(confidence) if isinstance(confidence, (int, float)) else None
        if isinstance(label, list):
            return SpanAttribute(
                name=name,
                label=[str(item) for item in label],
                confidence=conf_f,
            )
        return SpanAttribute(name=name, label=str(label or ""), confidence=conf_f)
    return SpanAttribute(name=name, label=str(value))


def parse_entities(raw: Any, text: str) -> list[ExtractedEntity]:
    if not isinstance(raw, dict):
        raise TaskExecutionError("GLiNER 2.5 returned an invalid entity result.")
    grouped = raw.get("entities", raw)
    if not isinstance(grouped, dict):
        raise TaskExecutionError("GLiNER 2.5 returned an invalid entity mapping.")
    entities: list[ExtractedEntity] = []
    for entity_type, items in grouped.items():
        for item in _as_list(items):
            span_text, start, end, confidence, extras = _span_item(item)
            if start is not None and end is not None and 0 <= start < end <= len(text):
                span_text = text[start:end]
            attributes = {
                name: _parse_attribute(str(name), value) for name, value in extras.items()
            }
            entities.append(
                ExtractedEntity(
                    entity_type=str(entity_type),
                    text=span_text,
                    start=start,
                    end=end,
                    confidence=confidence,
                    attributes=attributes,
                )
            )
    return entities


def parse_records(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(raw, dict):
        raise TaskExecutionError("GLiNER 2.5 returned an invalid record result.")
    records: dict[str, list[dict[str, Any]]] = {}
    for record_type, items in raw.items():
        if record_type in {"entities", "relation_extraction"}:
            continue
        parsed: list[dict[str, Any]] = []
        for item in _as_list(items):
            if isinstance(item, dict):
                parsed.append(item)
            else:
                parsed.append({"value": item})
        records[str(record_type)] = parsed
    return records


def _label_value(value: Any) -> str | list[str]:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        labels: list[str] = []
        for item in value:
            if isinstance(item, dict):
                labels.append(str(item.get("label", item.get("value", ""))))
            else:
                labels.append(str(item))
        return labels
    if isinstance(value, dict):
        label = value.get("label", value.get("value"))
        if isinstance(label, list):
            return [str(item) for item in label]
        if label is not None:
            return str(label)
    raise TaskExecutionError(f"GLiNER 2.5 returned an invalid label value: {value!r}.")


def parse_unconstrained_classification(
    text: str,
    raw: Any,
    *,
    model_id: str,
) -> SchemaClassification:
    if not isinstance(raw, dict):
        raise TaskExecutionError("GLiNER 2.5 returned an invalid classification result.")
    tasks: dict[str, TaskClassification] = {}
    for task_name, value in raw.items():
        confidence = None
        probabilities: dict[str, float] = {}
        if isinstance(value, dict):
            confidence_raw = value.get("confidence")
            if isinstance(confidence_raw, (int, float)):
                confidence = float(confidence_raw)
            probs = value.get("probabilities")
            if isinstance(probs, dict):
                probabilities = {
                    str(label): float(score)
                    for label, score in probs.items()
                    if isinstance(score, (int, float))
                }
        tasks[str(task_name)] = TaskClassification(
            task=str(task_name),
            value=_label_value(value),
            confidence=confidence,
            probabilities=probabilities,
        )
    return SchemaClassification(
        text=text,
        tasks=tasks,
        feasible=True,
        backend_used=BACKEND_NAME,
        model_id=model_id,
    )


def parse_constrained_classification(
    text: str,
    result: Any,
    *,
    model_id: str,
    task_names: Sequence[str],
) -> SchemaClassification:
    tasks: dict[str, TaskClassification] = {}
    for task_name in task_names:
        value = result.value(task_name)
        if isinstance(value, tuple):
            value = list(value)
        confidence = None
        confidence_fn = getattr(result, "confidence", None)
        if callable(confidence_fn):
            confidence_raw = confidence_fn(task_name)
            if isinstance(confidence_raw, (int, float)):
                confidence = float(confidence_raw)
        probabilities: dict[str, float] = {}
        probabilities_fn = getattr(result, "probabilities", None)
        if callable(probabilities_fn):
            probs = probabilities_fn(task_name)
            if isinstance(probs, dict):
                probabilities = {
                    str(label): float(score)
                    for label, score in probs.items()
                    if isinstance(score, (int, float))
                }
        tasks[task_name] = TaskClassification(
            task=task_name,
            value=cast(str | list[str], value),
            confidence=confidence,
            probabilities=probabilities,
        )
    feasible = bool(getattr(result, "feasible", True))
    return SchemaClassification(
        text=text,
        tasks=tasks,
        feasible=feasible,
        backend_used=BACKEND_NAME,
        model_id=model_id,
    )


def parse_graph(text: str, result: Any, *, model_id: str) -> GraphExtraction:
    entities: list[GraphEntity] = []
    for item in list(getattr(result, "entities", []) or []):
        entity_id = str(getattr(item, "id", ""))
        entity_type = str(getattr(item, "type", getattr(item, "entity_type", "")))
        start = getattr(item, "start", None)
        end = getattr(item, "end", None)
        confidence = getattr(item, "confidence", None)
        span_text = str(getattr(item, "text", ""))
        start_i = int(start) if isinstance(start, (int, float)) else None
        end_i = int(end) if isinstance(end, (int, float)) else None
        if start_i is not None and end_i is not None and 0 <= start_i < end_i <= len(text):
            span_text = text[start_i:end_i]
        entities.append(
            GraphEntity(
                id=entity_id,
                entity_type=entity_type,
                text=span_text,
                start=start_i,
                end=end_i,
                confidence=(
                    float(confidence) if isinstance(confidence, (int, float)) else None
                ),
            )
        )
    by_id = {entity.id: entity for entity in entities}
    relations: list[GraphRelation] = []
    for item in list(getattr(result, "relations", []) or []):
        head_id = str(getattr(item, "head", ""))
        tail_id = str(getattr(item, "tail", ""))
        confidence = getattr(item, "confidence", None)
        head = by_id.get(head_id)
        tail = by_id.get(tail_id)
        if head is None:
            looked_up = result.entity(head_id)
            head_text = str(getattr(looked_up, "text", ""))
        else:
            head_text = head.text
        if tail is None:
            looked_up = result.entity(tail_id)
            tail_text = str(getattr(looked_up, "text", ""))
        else:
            tail_text = tail.text
        relations.append(
            GraphRelation(
                relation_type=str(getattr(item, "type", getattr(item, "relation_type", ""))),
                head_id=head_id,
                tail_id=tail_id,
                head_text=head_text,
                tail_text=tail_text,
                confidence=(
                    float(confidence) if isinstance(confidence, (int, float)) else None
                ),
            )
        )
    return GraphExtraction(
        text=text,
        entities=entities,
        relations=relations,
        feasible=bool(getattr(result, "feasible", True)),
        backend_used=BACKEND_NAME,
        model_id=model_id,
    )


def _build_attribute_groups(attributes: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from gliner2 import AttributeGroup
    except ImportError as exc:
        raise RuntimeImportError(
            "Install 'aibackends[gliner25]' to use span attributes."
        ) from exc
    groups: dict[str, Any] = {}
    for name, spec in attributes.items():
        if hasattr(spec, "labels"):
            groups[str(name)] = spec
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"Attribute {name!r} must be a mapping or AttributeGroup.")
        groups[str(name)] = AttributeGroup(
            list(spec["labels"]),
            multi_label=bool(spec.get("multi_label", False)),
            threshold=float(spec.get("threshold", DEFAULT_THRESHOLD)),
            applies_to=(
                list(spec["applies_to"]) if spec.get("applies_to") is not None else None
            ),
            qualify_labels=bool(spec.get("qualify_labels", True)),
        )
    return groups


def _pair(value: Any) -> tuple[str, str]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return str(value[0]), str(value[1])
    raise ValueError(f"Constraint pair must be a two-item sequence, got {value!r}.")


def _build_constraints(constraints: Sequence[Mapping[str, Any]]) -> list[Any]:
    try:
        from gliner2.classification import constraints as constraint_dsl
    except ImportError as exc:
        raise RuntimeImportError(
            "Install 'aibackends[gliner25]' for constrained classification."
        ) from exc
    built: list[Any] = []
    for spec in constraints:
        kind = str(spec.get("type", "")).strip().lower()
        if kind == "implies":
            built.append(constraint_dsl.implies(_pair(spec["if"]), _pair(spec["then"])))
        elif kind == "excludes":
            built.append(
                constraint_dsl.excludes(_pair(spec["left"]), _pair(spec["right"]))
            )
        elif kind == "iff":
            built.append(constraint_dsl.iff(_pair(spec["if"]), _pair(spec["then"])))
        elif kind == "at_most":
            built.append(constraint_dsl.at_most(str(spec["task"]), int(spec["count"])))
        elif kind == "at_least":
            built.append(constraint_dsl.at_least(str(spec["task"]), int(spec["count"])))
        elif kind == "exactly":
            built.append(constraint_dsl.exactly(str(spec["task"]), int(spec["count"])))
        else:
            raise ValueError(
                "Unsupported constraint type. Use implies, excludes, iff, "
                "at_most, at_least, or exactly."
            )
    return built


def _build_classification_schema(
    tasks: Mapping[str, Mapping[str, Any] | Sequence[str]],
    constraints: Sequence[Mapping[str, Any]] | None,
) -> Any:
    try:
        from gliner2.classification import ClassificationSchema
    except ImportError as exc:
        raise RuntimeImportError(
            "Install 'aibackends[gliner25]' for constrained classification."
        ) from exc
    schema = ClassificationSchema()
    for name, spec in tasks.items():
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            schema = schema.single(str(name), list(spec))
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"Classification task {name!r} must be a mapping or list.")
        labels = spec.get("labels")
        if labels is None:
            raise ValueError(f"Classification task {name!r} is missing labels.")
        if spec.get("multi_label"):
            multi_kwargs: dict[str, Any] = {
                "min_labels": int(spec.get("min_labels", 0)),
            }
            if spec.get("max_labels") is not None:
                multi_kwargs["max_labels"] = int(spec["max_labels"])
            schema = schema.multi(
                str(name),
                list(labels) if not isinstance(labels, Mapping) else labels,
                **multi_kwargs,
            )
        else:
            schema = schema.single(
                str(name),
                list(labels) if not isinstance(labels, Mapping) else labels,
            )
    if constraints:
        schema = schema.constrain(*_build_constraints(constraints))
    return schema


def _classify_text_schema(
    tasks: Mapping[str, Mapping[str, Any] | Sequence[str]],
) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for name, spec in tasks.items():
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            schema[str(name)] = list(spec)
            continue
        if not isinstance(spec, Mapping):
            raise ValueError(f"Classification task {name!r} must be a mapping or list.")
        labels = spec.get("labels")
        if labels is None:
            raise ValueError(f"Classification task {name!r} is missing labels.")
        if spec.get("multi_label"):
            entry: dict[str, Any] = {
                "labels": list(labels) if not isinstance(labels, Mapping) else labels,
                "multi_label": True,
            }
            if "threshold" in spec:
                entry["cls_threshold"] = spec["threshold"]
            schema[str(name)] = entry
        else:
            schema[str(name)] = (
                list(labels) if not isinstance(labels, Mapping) else labels
            )
    return schema


def detect_pii_entities(
    text: str,
    labels: Sequence[str],
    *,
    model_id: str = DEFAULT_GLINER25_MODEL,
    device: str = "cpu",
    threshold: float = DEFAULT_THRESHOLD,
) -> list[PIIEntity]:
    backend = GLINER25_BACKEND
    extracted = backend.extract_entities(
        text,
        labels,
        device=device,
        model=model_id,
        threshold=threshold,
    )
    entities: list[PIIEntity] = []
    for item in extracted.entities:
        if item.start is None or item.end is None or item.end <= item.start:
            continue
        if item.start < 0 or item.end > len(text):
            continue
        entities.append(
            PIIEntity(
                entity_type=item.entity_type.upper().replace(" ", "_"),
                text=text[item.start : item.end],
                start=item.start,
                end=item.end,
                replacement="",
            )
        )
    return entities


class GLiNER25Backend(BaseExtractionBackend):
    name = BACKEND_NAME
    aliases = ("gliner-2.5", "gliner2.5")

    def load(self, *, device: str = "cpu", model: str | None = None) -> Any:
        return load_gliner25_extractor(resolve_gliner25_model_id(model), device)

    def extract_entities(
        self,
        text: str,
        labels: Sequence[str] | Mapping[str, str],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        long: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        attributes: Mapping[str, Any] | None = None,
    ) -> EntityExtraction:
        _validate_threshold(threshold)
        model_id = resolve_gliner25_model_id(model)
        extractor = load_gliner25_extractor(model_id, device)
        with _inference_lock(model_id, device):
            if attributes:
                schema = extractor.create_schema().entities(labels)
                schema = schema.entity_attributes(_build_attribute_groups(attributes))
                if long:
                    raw = extractor.extract_long(
                        text,
                        schema,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        include_spans=True,
                        include_confidence=True,
                        threshold=threshold,
                    )
                else:
                    raw = extractor.extract(
                        text,
                        schema,
                        include_spans=True,
                        include_confidence=True,
                        threshold=threshold,
                    )
            elif long:
                raw = extractor.extract_entities_long(
                    text,
                    labels,
                    threshold=threshold,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    include_spans=True,
                    include_confidence=True,
                )
            else:
                raw = extractor.extract_entities(
                    text,
                    labels,
                    threshold=threshold,
                    include_spans=True,
                    include_confidence=True,
                )
        return EntityExtraction(
            text=text,
            entities=parse_entities(raw, text),
            backend_used=self.name,
            model_id=model_id,
        )

    def extract_records(
        self,
        text: str,
        schema: Mapping[str, Any],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        long: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> RecordExtraction:
        _validate_threshold(threshold)
        model_id = resolve_gliner25_model_id(model)
        extractor = load_gliner25_extractor(model_id, device)
        structures = dict(schema)
        with _inference_lock(model_id, device):
            if long:
                raw = extractor.extract_json_long(
                    text,
                    structures,
                    threshold=threshold,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    include_spans=True,
                    include_confidence=True,
                )
            else:
                raw = extractor.extract_json(
                    text,
                    structures,
                    threshold=threshold,
                    include_spans=True,
                    include_confidence=True,
                )
        return RecordExtraction(
            text=text,
            records=parse_records(raw),
            backend_used=self.name,
            model_id=model_id,
        )

    def classify_schema(
        self,
        text: str,
        tasks: Mapping[str, Mapping[str, Any] | Sequence[str]],
        *,
        device: str = "cpu",
        model: str | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        constraints: Sequence[Mapping[str, Any]] | None = None,
    ) -> SchemaClassification:
        _validate_threshold(threshold)
        model_id = resolve_gliner25_model_id(model)
        if constraints:
            classifier = load_gliner25_classifier(model_id, device)
            schema = _build_classification_schema(tasks, constraints)
            with _inference_lock(model_id, device):
                result = classifier.classify(text, schema)
            return parse_constrained_classification(
                text,
                result,
                model_id=model_id,
                task_names=list(tasks),
            )
        extractor = load_gliner25_extractor(model_id, device)
        classify_schema = _classify_text_schema(tasks)
        with _inference_lock(model_id, device):
            raw = extractor.classify_text(
                text,
                classify_schema,
                threshold=threshold,
                include_confidence=True,
            )
        return parse_unconstrained_classification(text, raw, model_id=model_id)

    def extract_graph(
        self,
        text: str,
        entities: Sequence[str] | Mapping[str, str],
        relations: Sequence[Mapping[str, Any]],
        *,
        device: str = "cpu",
        model: str | None = None,
        no_self_loops: bool = True,
        long: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> GraphExtraction:
        model_id = resolve_gliner25_model_id(model)
        joint = load_gliner25_joint(model_id, device)
        schema = joint.create_schema().entities(entities)
        for relation in relations:
            name = str(relation["name"])
            schema = schema.relation(
                name,
                relation["head"],
                relation["tail"],
                unique_head=bool(relation.get("unique_head", False)),
                unique_tail=bool(relation.get("unique_tail", False)),
                allow_self=bool(relation.get("allow_self", False)),
            )
        if no_self_loops:
            schema = schema.no_self_loops()
        with _inference_lock(model_id, device):
            if long:
                result = joint.extract_long(
                    text,
                    schema,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:
                result = joint.extract(text, schema)
        return parse_graph(text, result, model_id=model_id)


GLINER25_BACKEND = GLiNER25Backend()

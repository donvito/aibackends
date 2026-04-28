from __future__ import annotations

from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import PIIBackendSpec, discover_specs, normalize_name, register_spec
from aibackends.schemas.pii import PIIEntity, RedactedText

_PII_BACKENDS: dict[str, PIIBackendSpec] = {}
_BUILTINS_REGISTERED = False


def register_pii_backend(spec: PIIBackendSpec) -> None:
    register_spec(_PII_BACKENDS, spec, spec.names)


def get_pii_backend(name: str) -> PIIBackendSpec:
    _ensure_builtin_pii_backends_registered()
    try:
        return _PII_BACKENDS[normalize_name(name)]
    except KeyError as exc:
        raise TaskExecutionError(f"Unsupported PII backend: {name}") from exc


def list_pii_backends() -> list[str]:
    _ensure_builtin_pii_backends_registered()
    return sorted({spec.name for spec in _PII_BACKENDS.values()})


def apply_redactions(
    text: str,
    entities: list[PIIEntity],
    *,
    backend_name: str,
) -> RedactedText:
    parts: list[str] = []
    cursor = 0
    redaction_map: dict[str, str] = {}
    normalized: list[PIIEntity] = []

    for index, entity in enumerate(sorted(entities, key=lambda item: item.start), start=1):
        if entity.start < cursor:
            continue
        replacement = f"[{entity.entity_type}_{index}]"
        parts.append(text[cursor : entity.start])
        parts.append(replacement)
        cursor = entity.end
        redaction_map[entity.text] = replacement
        normalized.append(entity.model_copy(update={"replacement": replacement}))

    parts.append(text[cursor:])
    return RedactedText(
        original_text=text,
        redacted_text="".join(parts),
        entities_found=normalized,
        redaction_map=redaction_map,
        backend_used=backend_name,
    )


def _ensure_builtin_pii_backends_registered() -> None:
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    for spec in discover_specs("aibackends.backends.pii", "PII_BACKEND_SPEC"):
        if not isinstance(spec, PIIBackendSpec):
            raise TaskExecutionError(f"Invalid PII backend spec: {spec!r}")
        register_pii_backend(spec)
    _BUILTINS_REGISTERED = True


__all__ = [
    "apply_redactions",
    "get_pii_backend",
    "list_pii_backends",
    "register_pii_backend",
]

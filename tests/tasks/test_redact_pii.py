from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest

from aibackends.core.exceptions import TaskExecutionError

gliner_module = importlib.import_module("aibackends.backends.pii.gliner")
redact_pii_module = importlib.import_module("aibackends.tasks.redact_pii")


class _FakeGLinerModel:
    def __init__(self, predictions: list[dict[str, Any]]) -> None:
        self.predictions = predictions
        self.calls: list[dict[str, Any]] = []

    def predict_entities(
        self,
        text: str,
        labels: list[str],
        *,
        threshold: float,
    ) -> list[dict[str, Any]]:
        self.calls.append({"text": text, "labels": list(labels), "threshold": threshold})
        return list(self.predictions)


@pytest.fixture(autouse=True)
def _reset_gliner_cache() -> Iterator[None]:
    gliner_module.clear_model_cache()
    yield
    gliner_module.clear_model_cache()


def _install_fake_model(
    monkeypatch: pytest.MonkeyPatch,
    predictions: list[dict[str, Any]],
) -> _FakeGLinerModel:
    fake = _FakeGLinerModel(predictions)
    monkeypatch.setitem(
        gliner_module._MODEL_CACHE,
        gliner_module.GLINER_MODEL_ID,
        fake,
    )
    return fake


def test_redact_pii_passes_through_when_no_entities(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_model(monkeypatch, [])

    result = redact_pii_module.redact_pii(
        "Email john@example.com or call +1 555 010 9999", backend="gliner"
    )

    assert result.redacted_text == "Email john@example.com or call +1 555 010 9999"
    assert result.entities_found == []
    assert result.redaction_map == {}
    assert result.backend_used == "gliner"


def test_redact_pii_uses_nvidia_gliner_model(monkeypatch: pytest.MonkeyPatch) -> None:
    text = (
        "Hi support, I can't log in! My account username is 'johndoe88'. "
        "Every time I try, it says 'invalid credentials'. Please reset my password. "
        "You can reach me at (555) 123-4567 or johnd@example.com"
    )
    username = "johndoe88"
    phone = "(555) 123-4567"
    email = "johnd@example.com"
    fake = _install_fake_model(
        monkeypatch,
        [
            {
                "start": text.index(username),
                "end": text.index(username) + len(username),
                "label": "user_name",
            },
            {
                "start": text.index(phone),
                "end": text.index(phone) + len(phone),
                "label": "phone_number",
            },
            {
                "start": text.index(email),
                "end": text.index(email) + len(email),
                "label": "email",
            },
        ],
    )

    result = redact_pii_module.redact_pii(text, backend="gliner")

    assert fake.calls == [
        {
            "text": text,
            "labels": list(gliner_module.GLINER_LABELS),
            "threshold": gliner_module.GLINER_THRESHOLD,
        }
    ]
    assert result.backend_used == "gliner"
    assert any(entity.entity_type == "USER_NAME" for entity in result.entities_found)
    assert "USER_NAME" in result.redacted_text


def test_redact_pii_accepts_custom_gliner_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _install_fake_model(monkeypatch, [])

    redact_pii_module.redact_pii(
        "Contact me at john@example.com",
        backend="gliner",
        labels=["email", "user_name"],
    )

    assert fake.calls == [
        {
            "text": "Contact me at john@example.com",
            "labels": ["email", "user_name"],
            "threshold": gliner_module.GLINER_THRESHOLD,
        }
    ]


def test_get_pii_backend_redact_method_uses_native_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aibackends.backends.pii import get_pii_backend

    text = "Reach me at jane@example.com"
    fake = _install_fake_model(
        monkeypatch,
        [
            {
                "start": text.index("jane@example.com"),
                "end": text.index("jane@example.com") + len("jane@example.com"),
                "label": "email",
            }
        ],
    )

    backend = get_pii_backend("gliner")
    result = backend.redact(text, labels=["email"])

    assert result.backend_used == "gliner"
    assert "[EMAIL_1]" in result.redacted_text
    assert fake.calls == [
        {"text": text, "labels": ["email"], "threshold": gliner_module.GLINER_THRESHOLD}
    ]


def test_get_pii_backend_load_warms_native_model(monkeypatch: pytest.MonkeyPatch) -> None:
    from aibackends.backends.pii import get_pii_backend

    fake = _install_fake_model(monkeypatch, [])
    backend = get_pii_backend("gliner")

    assert backend.load() is fake
    assert fake.calls == []


def test_load_gliner_model_caches_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    load_count = {"value": 0}

    class _Stub:
        @classmethod
        def from_pretrained(cls, model_id: str) -> _FakeGLinerModel:
            del model_id
            load_count["value"] += 1
            return _FakeGLinerModel([])

    fake_module = ModuleType("gliner")
    fake_module.GLiNER = _Stub  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gliner", fake_module)

    spec = gliner_module.PII_BACKEND_SPEC
    first = gliner_module.load_gliner_model(spec)
    second = gliner_module.load_gliner_model(spec)

    assert first is second
    assert load_count["value"] == 1


def test_redact_pii_rejects_unknown_backend() -> None:
    with pytest.raises(TaskExecutionError, match="Unsupported PII backend"):
        redact_pii_module.redact_pii("hello", backend="regex")


def test_redact_pii_rejects_custom_labels_for_openai_privacy() -> None:
    with pytest.raises(TaskExecutionError, match="Custom labels are not supported"):
        redact_pii_module.redact_pii("hello", backend="openai-privacy", labels=["email"])

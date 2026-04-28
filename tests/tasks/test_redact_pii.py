from __future__ import annotations

import importlib

import pytest

from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError

gliner_module = importlib.import_module("aibackends.backends.pii.gliner")
redact_pii_module = importlib.import_module("aibackends.tasks.redact_pii")


@pytest.fixture(autouse=True)
def clear_gliner_cache():
    gliner_module._MODEL_CACHE.clear()
    yield
    gliner_module._MODEL_CACHE.clear()


def test_redact_pii_does_not_fallback_to_regex(monkeypatch):
    class FakeModel:
        def predict_entities(self, text: str, labels: list[str], threshold: float):
            del text, labels, threshold
            return []

    monkeypatch.setattr(gliner_module, "_get_model", lambda model_id, device=None: FakeModel())

    result = redact_pii_module.redact_pii(
        "Email john@example.com or call +1 555 010 9999", backend="gliner"
    )
    assert result.redacted_text == "Email john@example.com or call +1 555 010 9999"
    assert result.entities_found == []
    assert result.redaction_map == {}
    assert result.backend_used == "gliner"


def test_redact_pii_uses_nvidia_gliner_model(monkeypatch):
    text = (
        "Hi support, I can't log in! My account username is 'johndoe88'. "
        "Every time I try, it says 'invalid credentials'. Please reset my password. "
        "You can reach me at (555) 123-4567 or johnd@example.com"
    )
    captured: dict[str, object] = {"from_pretrained_calls": 0}

    username = "johndoe88"
    phone = "(555) 123-4567"
    email = "johnd@example.com"

    class FakeModel:
        def predict_entities(self, payload_text: str, labels: list[str], threshold: float):
            captured["predict_text"] = payload_text
            captured["predict_labels"] = labels
            captured["predict_threshold"] = threshold
            return [
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
            ]

    class FakeGLiNER:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object):
            captured["from_pretrained_calls"] = int(captured["from_pretrained_calls"]) + 1
            captured["model_id"] = model_id
            captured["load_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(gliner_module, "_load_gliner_class", lambda: FakeGLiNER)

    result = redact_pii_module.redact_pii(text, backend="gliner")

    assert captured["from_pretrained_calls"] == 1
    assert captured["model_id"] == gliner_module.GLINER_MODEL_ID
    assert captured["load_kwargs"] == {}
    assert captured["predict_text"] == text
    assert captured["predict_labels"] == list(gliner_module.GLINER_LABELS)
    assert captured["predict_threshold"] == gliner_module.GLINER_THRESHOLD
    assert result.backend_used == "gliner"
    assert any(entity.entity_type == "USER_NAME" for entity in result.entities_found)
    assert "USER_NAME" in result.redacted_text


def test_redact_pii_accepts_custom_gliner_labels(monkeypatch):
    captured: dict[str, object] = {}

    class FakeModel:
        def predict_entities(self, text: str, labels: list[str], threshold: float):
            del text, threshold
            captured["labels"] = labels
            return []

    monkeypatch.setattr(gliner_module, "_get_model", lambda model_id, device=None: FakeModel())

    redact_pii_module.redact_pii(
        "Contact me at john@example.com", backend="gliner", labels=["email", "user_name"]
    )

    assert captured["labels"] == ["email", "user_name"]


def test_gliner_backend_reuses_cached_model(monkeypatch):
    captured: dict[str, int] = {"from_pretrained_calls": 0}

    class FakeModel:
        def predict_entities(self, text: str, labels: list[str], threshold: float):
            del text, labels, threshold
            return []

    class FakeGLiNER:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object):
            del model_id, kwargs
            captured["from_pretrained_calls"] += 1
            return FakeModel()

    monkeypatch.setattr(gliner_module, "_load_gliner_class", lambda: FakeGLiNER)

    redact_pii_module.redact_pii("first", backend="gliner")
    redact_pii_module.redact_pii("second", backend="gliner")

    assert captured["from_pretrained_calls"] == 1


def test_redact_pii_passes_cuda_device_to_gliner(monkeypatch):
    captured: dict[str, object] = {}

    class FakeModel:
        def predict_entities(self, text: str, labels: list[str], threshold: float):
            del text, labels, threshold
            return []

    class FakeGLiNER:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object):
            captured["model_id"] = model_id
            captured["load_kwargs"] = kwargs
            return FakeModel()

    monkeypatch.setattr(gliner_module, "_load_gliner_class", lambda: FakeGLiNER)

    redact_pii_module.redact_pii("hello", backend="gliner", device="cuda")

    assert captured["model_id"] == gliner_module.GLINER_MODEL_ID
    assert captured["load_kwargs"] == {"map_location": "cuda"}


def test_gliner_cache_separates_cpu_and_cuda_models(monkeypatch):
    captured: dict[str, int] = {"from_pretrained_calls": 0}

    class FakeModel:
        def predict_entities(self, text: str, labels: list[str], threshold: float):
            del text, labels, threshold
            return []

    class FakeGLiNER:
        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object):
            del model_id, kwargs
            captured["from_pretrained_calls"] += 1
            return FakeModel()

    monkeypatch.setattr(gliner_module, "_load_gliner_class", lambda: FakeGLiNER)

    redact_pii_module.redact_pii("cpu", backend="gliner")
    redact_pii_module.redact_pii("cuda", backend="gliner", device="cuda")

    assert captured["from_pretrained_calls"] == 2


def test_gliner_backend_reports_missing_dependency(monkeypatch):
    def fail_import():
        raise RuntimeImportError("Install 'aibackends[pii]' to use the GLiNER PII backend.")

    monkeypatch.setattr(gliner_module, "_load_gliner_class", fail_import)

    with pytest.raises(RuntimeImportError, match="Install 'aibackends\\[pii\\]'"):
        gliner_module._get_model(gliner_module.GLINER_MODEL_ID)


def test_redact_pii_rejects_unknown_backend():
    with pytest.raises(TaskExecutionError, match="Unsupported PII backend"):
        redact_pii_module.redact_pii("hello", backend="regex")


def test_redact_pii_rejects_custom_labels_for_openai_privacy():
    with pytest.raises(TaskExecutionError, match="Custom labels are not supported"):
        redact_pii_module.redact_pii("hello", backend="openai-privacy", labels=["email"])

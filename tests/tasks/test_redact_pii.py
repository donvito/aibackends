from __future__ import annotations

import importlib
import json

import pytest

from aibackends.core.exceptions import TaskExecutionError

redact_pii_module = importlib.import_module("aibackends.tasks.redact_pii")


def test_redact_pii_does_not_fallback_to_regex(monkeypatch):
    monkeypatch.setattr(redact_pii_module, "_gliner_entities", lambda text, labels=None: [])

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
    captured: dict[str, object] = {}

    username = "johndoe88"
    phone = "(555) 123-4567"
    email = "johnd@example.com"

    class CompletedProcess:
        returncode = 0
        stdout = json.dumps(
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
            ]
        )

    def fake_run(command, *, input: str, text: bool, capture_output: bool, check: bool):
        payload = json.loads(input)
        captured["command"] = command
        captured["payload"] = payload
        captured["text"] = text
        captured["capture_output"] = capture_output
        captured["check"] = check
        return CompletedProcess()

    monkeypatch.setattr(redact_pii_module.subprocess, "run", fake_run)

    result = redact_pii_module.redact_pii(text, backend="gliner")

    assert captured["command"] == [
        redact_pii_module.sys.executable,
        str(redact_pii_module.GLINER_WORKER_PATH),
    ]
    assert captured["payload"] == {
        "model_id": redact_pii_module.GLINER_MODEL_ID,
        "text": text,
        "labels": redact_pii_module.GLINER_LABELS,
        "threshold": redact_pii_module.GLINER_THRESHOLD,
    }
    assert captured["text"] is True
    assert captured["capture_output"] is True
    assert captured["check"] is False
    assert result.backend_used == "gliner"
    assert any(entity.entity_type == "USER_NAME" for entity in result.entities_found)
    assert "USER_NAME" in result.redacted_text


def test_redact_pii_accepts_custom_gliner_labels(monkeypatch):
    captured: dict[str, object] = {}

    class CompletedProcess:
        returncode = 0
        stdout = "[]"

    def fake_run(command, *, input: str, text: bool, capture_output: bool, check: bool):
        del command, text, capture_output, check
        captured["payload"] = json.loads(input)
        return CompletedProcess()

    monkeypatch.setattr(redact_pii_module.subprocess, "run", fake_run)

    redact_pii_module.redact_pii(
        "Contact me at john@example.com", backend="gliner", labels=["email", "user_name"]
    )

    assert captured["payload"] == {
        "model_id": redact_pii_module.GLINER_MODEL_ID,
        "text": "Contact me at john@example.com",
        "labels": ["email", "user_name"],
        "threshold": redact_pii_module.GLINER_THRESHOLD,
    }


def test_redact_pii_rejects_unknown_backend():
    with pytest.raises(TaskExecutionError, match="Unsupported PII backend"):
        redact_pii_module.redact_pii("hello", backend="regex")


def test_redact_pii_rejects_custom_labels_for_openai_privacy():
    with pytest.raises(TaskExecutionError, match="Custom labels are only supported"):
        redact_pii_module.redact_pii("hello", backend="openai-privacy", labels=["email"])

from __future__ import annotations

import importlib
from typing import Any

import pytest
from typer.testing import CliRunner

from aibackends.cli import app
from aibackends.schemas.moderation import ResponseModeration

runner = CliRunner()
moderation_module = importlib.import_module("aibackends.tasks.moderation")


def test_task_command_accepts_runtime_and_model_strings():
    result = runner.invoke(
        app,
        [
            "task",
            "summarize",
            "--input",
            "Meeting notes",
            "--runtime",
            "stub",
            "--model",
            "stub-model",
        ],
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "Stub summary"


def test_check_command_accepts_runtime_and_model_strings():
    result = runner.invoke(
        app,
        [
            "check",
            "stub",
            "--model",
            "stub-model",
        ],
    )

    assert result.exit_code == 0
    assert '"runtime": "stub"' in result.stdout
    assert '"client": "StubRuntime"' in result.stdout
    assert '"model": "stub-model"' in result.stdout


def test_task_command_supports_gliguard_response_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class _Backend:
        def moderate_response(
            self,
            response: str,
            **kwargs: Any,
        ) -> ResponseModeration:
            captured["response"] = response
            captured.update(kwargs)
            return ResponseModeration(
                response=response,
                prompt=kwargs["prompt"],
                is_safe=False,
                safety="unsafe",
                toxicity=["non_violent_crime"],
                refusal="compliance",
                backend_used="gliguard",
                model_id="fastino/gliguard-LLMGuardrails-300M",
            )

    monkeypatch.setattr(moderation_module, "get_moderation_backend", lambda name: _Backend())
    result = runner.invoke(
        app,
        [
            "task",
            "moderate-response",
            "--input",
            "A harmful answer",
            "--prompt",
            "A harmful request",
            "--backend",
            "gliguard",
            "--device",
            "gpu",
            "--threshold",
            "0.6",
            "--category-threshold",
            "0.3",
        ],
    )

    assert result.exit_code == 0
    assert '"safety": "unsafe"' in result.stdout
    assert captured == {
        "response": "A harmful answer",
        "prompt": "A harmful request",
        "device": "gpu",
        "threshold": 0.6,
        "category_threshold": 0.3,
    }


def test_task_command_supports_gliner25_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aibackends.schemas.extraction import EntityExtraction

    captured: dict[str, Any] = {}
    extraction_module = importlib.import_module("aibackends.tasks.extraction")

    def _extract_entities(text: str, **kwargs: Any) -> EntityExtraction:
        captured["text"] = text
        captured.update(kwargs)
        return EntityExtraction(
            text=text,
            entities=[],
            backend_used="gliner25",
            model_id="fastino/gliner2.5-small-v1",
        )

    monkeypatch.setattr(extraction_module, "extract_entities", _extract_entities)
    result = runner.invoke(
        app,
        [
            "task",
            "extract-entities",
            "--input",
            "Ada lives in London.",
            "--labels",
            "person,location",
            "--device",
            "cpu",
            "--model",
            "gliner25-small",
        ],
    )

    assert result.exit_code == 0
    assert captured["text"] == "Ada lives in London."
    assert captured["labels"] == ["person", "location"]
    assert captured["device"] == "cpu"
    assert captured["model"] == "gliner25-small"

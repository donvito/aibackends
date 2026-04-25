from __future__ import annotations

import importlib
from typing import Any, cast

import pytest
from pydantic import BaseModel

from aibackends.core.registry import ModelRef
from aibackends.runtimes import get_runtime_spec
from aibackends.tasks import classify, embed, extract, summarize
from aibackends.tasks._utils import load_text_input

classify_module = importlib.import_module("aibackends.tasks.classify")


class Person(BaseModel):
    name: str
    email: str


def test_summarize_returns_text():
    result = summarize("This is a long meeting transcript.")
    assert result == "Stub summary"


def test_classify_returns_typed_output():
    result = classify("invoice for April services", labels=["invoice", "contract", "receipt"])
    assert result.label == "invoice"
    assert result.all_scores["invoice"] > result.all_scores["contract"]


def test_classify_supports_label_descriptions_and_custom_prompts(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_structured_task(
        *,
        task_name,
        schema,
        messages,
        runtime=None,
        model=None,
        **overrides,
    ):
        captured["task_name"] = task_name
        captured["schema"] = schema
        captured["messages"] = messages
        captured["runtime"] = runtime
        captured["model"] = model
        captured["overrides"] = overrides
        return schema(
            label="ml-engineer",
            confidence=0.82,
            all_scores={"ml-engineer": 0.82, "data-engineer": 0.18},
        )

    monkeypatch.setattr(classify_module, "run_structured_task", fake_run_structured_task)

    result = classify_module.classify(
        "Built production ML systems and data pipelines.",
        labels=["ml-engineer", "data-engineer"],
        label_descriptions={
            "ml-engineer": "Owns model training, evaluation, and production inference.",
            "data-engineer": "Owns ETL pipelines, warehousing, and analytics plumbing.",
        },
        system_prompt="You are a recruiter matching resumes to roles.",
        prompt="Choose the single best-fitting role based only on resume evidence.",
        runtime=get_runtime_spec("stub"),
        model=ModelRef(name="stub-model"),
        max_retries=2,
    )

    assert result.label == "ml-engineer"
    assert captured["task_name"] == "classify"
    assert captured["runtime"] == get_runtime_spec("stub")
    assert captured["model"] == ModelRef(name="stub-model")
    assert captured["overrides"] == {"max_retries": 2}
    messages = cast(list[dict[str, Any]], captured["messages"])
    assert messages[0]["content"] == "You are a recruiter matching resumes to roles."
    assert (
        "Choose the single best-fitting role based only on resume evidence."
        in messages[1]["content"]
    )
    assert "Label descriptions:" in messages[1]["content"]
    assert (
        "- ml-engineer: Owns model training, evaluation, and production inference."
        in messages[1]["content"]
    )
    assert (
        "- data-engineer: Owns ETL pipelines, warehousing, and analytics plumbing."
        in messages[1]["content"]
    )
    assert "Text:\nBuilt production ML systems and data pipelines." in messages[1]["content"]


def test_classify_rejects_unknown_label_descriptions():
    with pytest.raises(ValueError, match="unknown labels: contract"):
        classify_module.classify(
            "invoice for April services",
            labels=["invoice", "receipt"],
            label_descriptions={"contract": "Agreement between parties."},
        )


def test_summarize_rejects_string_runtime_and_model_refs():
    with pytest.raises(TypeError, match="runtime must be a RuntimeSpec"):
        summarize("notes", runtime="stub")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="model must be a ModelRef"):
        summarize("notes", model="stub-model")  # type: ignore[arg-type]


def test_extract_uses_custom_schema():
    result = extract("Alice can be reached at alice@example.com", schema=Person)
    assert result.name == "Alice"
    assert result.email == "alice@example.com"


def test_embed_returns_vector():
    vector = embed("hello")
    assert vector == [5.0, 1.0, 0.5]


def test_load_text_input_treats_invalid_path_like_strings_as_text():
    value = "x" * 4096

    assert load_text_input(value) == value

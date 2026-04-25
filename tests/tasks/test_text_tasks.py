from __future__ import annotations

from pydantic import BaseModel

from aibackends.tasks import classify, embed, extract, summarize
from aibackends.tasks._utils import load_text_input


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

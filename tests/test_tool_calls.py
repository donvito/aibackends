from __future__ import annotations

from aibackends.core.tool_calls import (
    ToolCall,
    clean_answer,
    extract_tool_calls,
    strip_reasoning,
)


def test_extract_tool_calls_with_special_token_markers() -> None:
    content = (
        "Reasoning about the request.</think>"
        '<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>'
        "Checking the weather now."
    )

    calls = extract_tool_calls(content)

    assert calls == [ToolCall(name="get_weather", arguments={"city": "Paris"})]


def test_extract_tool_calls_without_markers_takes_last_match() -> None:
    content = (
        'I could call [get_weather(city="Rome")] here, but the user asked '
        'about Paris, so: [get_weather(city="Paris")]'
    )

    calls = extract_tool_calls(content)

    assert calls == [ToolCall(name="get_weather", arguments={"city": "Paris"})]


def test_extract_tool_calls_parses_multiple_calls_and_types() -> None:
    content = (
        "<|tool_call_start|>[get_weather(city=\"Rome\"), "
        "convert_currency(amount=50.5, from_currency='USD', "
        "to_currency='EUR')]<|tool_call_end|>"
    )

    calls = extract_tool_calls(content)

    assert calls == [
        ToolCall(name="get_weather", arguments={"city": "Rome"}),
        ToolCall(
            name="convert_currency",
            arguments={
                "amount": 50.5,
                "from_currency": "USD",
                "to_currency": "EUR",
            },
        ),
    ]


def test_extract_tool_calls_returns_empty_for_plain_answers() -> None:
    assert extract_tool_calls("Paris is the capital of France.") == []
    assert extract_tool_calls("Some [bracketed] prose without calls.") == []
    assert extract_tool_calls("") == []


def test_extract_tool_calls_ignores_invalid_python() -> None:
    assert extract_tool_calls("<|tool_call_start|>[get_weather(]<|tool_call_end|>") == []


def test_strip_reasoning_removes_think_block() -> None:
    assert strip_reasoning("thinking...</think>the answer") == "the answer"
    assert strip_reasoning("no reasoning here") == "no reasoning here"


def test_clean_answer_removes_markers_and_special_tokens() -> None:
    content = (
        "reasoning</think><|tool_call_start|>[f(x=1)]<|tool_call_end|>"
        "The answer.<|im_end|>"
    )

    assert clean_answer(content) == "The answer."

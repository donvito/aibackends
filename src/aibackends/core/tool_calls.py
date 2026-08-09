"""Helpers for parsing Pythonic tool calls from model responses.

LFM2.5-style models emit tool calls as a Python list between
`<|tool_call_start|>` and `<|tool_call_end|>` special tokens, e.g.
`<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>`. Some
runtimes strip special tokens during detokenization, so these helpers also
recognise bare Pythonic call lists in the response text.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[a-z_]+\|>")
TOOL_CALL_MARKERS_PATTERN = re.compile(
    r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL
)
PYTHONIC_CALL_PATTERN = re.compile(r"\[\s*[A-Za-z_]\w*\(.*?\)\s*\]", re.DOTALL)


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


def strip_reasoning(content: str) -> str:
    """Drop the `<think>...</think>` reasoning block reasoning models emit."""
    if "</think>" in content:
        return content.rsplit("</think>", 1)[1]
    return content


def extract_tool_calls(content: str) -> list[ToolCall]:
    """Parse Pythonic tool calls from a model response.

    Handles both raw output (with `<|tool_call_start|>` markers) and output
    where special tokens were stripped during detokenization. When markers
    are missing, the last bare Pythonic call list is used, since reasoning
    text may draft calls before the actual one.
    """
    text = strip_reasoning(content)
    marker_match = TOOL_CALL_MARKERS_PATTERN.search(text)
    if marker_match:
        call_text = marker_match.group(1).strip()
    else:
        bare_matches = PYTHONIC_CALL_PATTERN.findall(text)
        if not bare_matches:
            return []
        call_text = bare_matches[-1]

    try:
        parsed = ast.parse(call_text, mode="eval")
    except SyntaxError:
        return []
    if not isinstance(parsed.body, ast.List):
        return []

    calls: list[ToolCall] = []
    for node in parsed.body.elts:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        arguments: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                arguments[keyword.arg] = ast.literal_eval(keyword.value)
            except ValueError:
                arguments[keyword.arg] = None
        calls.append(ToolCall(name=node.func.id, arguments=arguments))
    return calls


def clean_answer(content: str) -> str:
    """Return the readable answer text without reasoning or special tokens."""
    text = strip_reasoning(content)
    text = TOOL_CALL_MARKERS_PATTERN.sub("", text)
    text = SPECIAL_TOKEN_PATTERN.sub("", text)
    return text.strip()

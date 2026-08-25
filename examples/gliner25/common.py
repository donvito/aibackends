"""Shared model, device, output, and span helpers for GLiNER2.5 examples."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping
from typing import Any, cast

MODEL_IDS = {
    "small": "fastino/gliner2.5-small-v1",
    "base": "fastino/gliner2.5-base-v1",
    "multi": "fastino/gliner2.5-multi-v1",
}
DEFAULT_MODEL = "small"


def add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_model: str = DEFAULT_MODEL,
) -> None:
    """Add the model and device flags shared by every runnable example."""
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_IDS),
        default=default_model,
        help="GLiNER2.5 checkpoint alias (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, gpu, cuda, cuda:<index>, or mps.",
    )


def resolve_model_id(model: str) -> str:
    """Resolve a short model alias or accept a full Hugging Face model ID."""
    value = model.strip()
    if value in MODEL_IDS:
        return MODEL_IDS[value]
    if value in MODEL_IDS.values():
        return value
    choices = ", ".join(MODEL_IDS)
    raise ValueError(f"Unknown GLiNER2.5 model {model!r}. Use one of: {choices}.")


def normalize_device(device: str) -> str:
    """Normalize explicit devices and auto-detect a local accelerator."""
    value = device.strip().lower()
    if value == "auto":
        try:
            import torch
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if value == "gpu":
        return "cuda"
    if value in {"cpu", "cuda", "mps"}:
        return value
    prefix, separator, index = value.partition(":")
    if prefix == "cuda" and separator and index.isdigit():
        return value
    raise ValueError(
        "Unsupported device. Use 'auto', 'cpu', 'gpu', 'cuda', 'cuda:<index>', or 'mps'."
    )


def load_extractor(model: str, device: str) -> tuple[Any, str, str]:
    """Load a GLiNER2.5 boundary checkpoint through the architecture-aware loader."""
    from gliner2 import AutoExtractor

    model_id = resolve_model_id(model)
    device_name = normalize_device(device)
    extractor = AutoExtractor.from_pretrained(model_id, map_location=device_name)
    evaluate = getattr(extractor, "eval", None)
    if callable(evaluate):
        evaluate()
    return extractor, model_id, device_name


def result_to_dict(result: object) -> dict[str, Any]:
    """Convert GLiNER2 dictionaries and typed decoder results to a dictionary."""
    if isinstance(result, dict):
        return cast(dict[str, Any], result)
    for method_name in ("to_dict", "model_dump"):
        method = getattr(result, method_name, None)
        if callable(method):
            value = method()
            if isinstance(value, dict):
                return cast(dict[str, Any], value)
    raise TypeError(f"Unsupported GLiNER2 result type: {type(result).__name__}.")


def print_result(result: object) -> None:
    """Print a GLiNER2 result as stable, readable JSON."""
    print(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False, sort_keys=True))


def iter_spans(value: object) -> Iterator[Mapping[str, object]]:
    """Yield every nested object carrying text and half-open character offsets."""
    if isinstance(value, Mapping):
        if {"text", "start", "end"}.issubset(value):
            yield value
            return
        for child in value.values():
            yield from iter_spans(child)
    elif isinstance(value, list | tuple):
        for child in value:
            yield from iter_spans(child)


def assert_source_spans(text: str, result: object) -> int:
    """Assert all returned spans point to their exact source substring."""
    count = 0
    for span in iter_spans(result_to_dict(result)):
        start = span["start"]
        end = span["end"]
        extracted = span["text"]
        if not isinstance(start, int) or not isinstance(end, int):
            raise AssertionError(f"Non-integer span offsets: {span!r}")
        if not isinstance(extracted, str):
            raise AssertionError(f"Non-string span text: {span!r}")
        if start < 0 or end <= start or end > len(text):
            raise AssertionError(f"Out-of-range span: {span!r}")
        actual = text[start:end]
        if actual != extracted:
            raise AssertionError(
                f"Span mismatch at [{start}:{end}]: expected {extracted!r}, found {actual!r}."
            )
        count += 1
    return count

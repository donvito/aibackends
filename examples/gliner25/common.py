"""Shared model, device, output, and span helpers for GLiNER2.5 examples."""

from __future__ import annotations

import argparse
import json

from aibackends.backends.information_extraction import (
    BaseInformationExtractionBackend,
    get_information_extraction_backend,
)
from aibackends.backends.information_extraction.gliner25 import (
    GLINER25_MODEL_IDS,
    assert_source_spans,
    normalize_device,
    resolve_model_id,
    result_to_dict,
)

MODEL_IDS = GLINER25_MODEL_IDS
DEFAULT_MODEL = "small"

__all__ = [
    "DEFAULT_MODEL",
    "MODEL_IDS",
    "add_model_arguments",
    "assert_source_spans",
    "load_backend",
    "normalize_device",
    "print_result",
    "resolve_model_id",
    "result_to_dict",
]


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


def load_backend(
    model: str,
    device: str,
) -> tuple[BaseInformationExtractionBackend, str, str]:
    """Preload the aibackends GLiNER2.5 backend and return resolved settings."""
    model_id = resolve_model_id(model)
    device_name = normalize_device(device)
    backend = get_information_extraction_backend("gliner25")
    backend.load(model=model, device=device_name)
    return backend, model_id, device_name


def print_result(result: object) -> None:
    """Print a GLiNER2 result as stable, readable JSON."""
    print(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False, sort_keys=True))

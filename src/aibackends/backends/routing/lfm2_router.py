"""Zero-shot prompt routing backed by LiquidAI's LFM2.5 encoder fine-tune.

The model (`LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`) is a bidirectional
LFM2.5 encoder with a zero-shot routing head. Its custom remote code exposes
`model.route(text, routes, tokenizer=...)`, which scores the text against
every free-text route in a single forward pass and returns a score-sorted
list of `{"route": ..., "score": ...}` dicts (softmax over the routes).
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from aibackends.backends.routing._base import BaseRoutingBackend
from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.schemas.routing import RouteScore, RoutingResult

ROUTER_MODEL_ID = "LiquidAI/LFM2.5-Encoder-350M-Prompt-Router"

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_INFERENCE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def normalize_device(device: str | None) -> str:
    normalized = (device or "cpu").strip().lower()
    if normalized == "gpu":
        return "cuda"
    if normalized in {"cpu", "cuda", "mps"}:
        return normalized
    prefix, separator, index = normalized.partition(":")
    if prefix == "cuda" and separator and index.isdigit():
        return normalized
    raise ValueError(
        "Unsupported prompt router device. Use 'cpu', 'gpu', 'cuda', 'cuda:<index>', or 'mps'."
    )


def load_router_model(device: str = "cpu") -> tuple[Any, Any]:
    """Load the router tokenizer and model once per process and device."""
    device_name = normalize_device(device)
    cache_key = (ROUTER_MODEL_ID, device_name)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            import torch  # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[routing]' to use the LFM2.5 prompt router backend."
            ) from exc

        # The routing head and its `route()` helper live in the model repo's
        # custom code, hence trust_remote_code on both loads.
        try:
            tokenizer = AutoTokenizer.from_pretrained(ROUTER_MODEL_ID, trust_remote_code=True)
        except ValueError:
            # The repo's tokenizer_config.json names a transformers v5
            # tokenizer class; on 4.x, load the fast tokenizer directly
            # (route() only needs offset mappings, which it supports).
            from transformers import PreTrainedTokenizerFast

            tokenizer = PreTrainedTokenizerFast.from_pretrained(ROUTER_MODEL_ID)
        model = AutoModel.from_pretrained(ROUTER_MODEL_ID, trust_remote_code=True)
        model = model.to(device_name)
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        entry = (tokenizer, model)
        _MODEL_CACHE[cache_key] = entry
        _INFERENCE_LOCKS[cache_key] = threading.Lock()
        return entry


def clear_model_cache() -> None:
    """Drop cached router models. Intended for tests and memory management."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _INFERENCE_LOCKS.clear()


def _validate_routes(routes: Sequence[str]) -> list[str]:
    if not routes:
        raise ValueError("routes must contain at least one label.")
    cleaned: list[str] = []
    for route in routes:
        if not isinstance(route, str) or not route.strip():
            raise ValueError("routes must be non-empty strings.")
        cleaned.append(route.strip())
    return cleaned


def _validate_threshold(threshold: float | None) -> None:
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")


def _inference_lock(device_name: str) -> threading.Lock:
    cache_key = (ROUTER_MODEL_ID, device_name)
    with _CACHE_LOCK:
        return _INFERENCE_LOCKS.setdefault(cache_key, threading.Lock())


def _route_text(
    text: str,
    routes: Sequence[str],
    *,
    device: str,
    threshold: float | None,
) -> list[Any]:
    device_name = normalize_device(device)
    tokenizer, model = load_router_model(device_name)
    with _inference_lock(device_name):
        raw = model.route(text, list(routes), tokenizer=tokenizer, threshold=threshold)
    if not isinstance(raw, list):
        raise TaskExecutionError("Prompt router returned an invalid routing result.")
    return raw


def _routing_result(text: str, raw: list[Any]) -> RoutingResult:
    scores: list[RouteScore] = []
    for item in raw:
        if not isinstance(item, dict):
            raise TaskExecutionError(f"Prompt router returned an invalid route entry: {item!r}.")
        route = item.get("route")
        score = item.get("score")
        if not isinstance(route, str) or not isinstance(score, int | float):
            raise TaskExecutionError(f"Prompt router returned an invalid route entry: {item!r}.")
        scores.append(RouteScore(route=route, score=float(score)))
    scores.sort(key=lambda entry: entry.score, reverse=True)
    return RoutingResult(
        text=text,
        best_route=scores[0].route if scores else None,
        scores=scores,
        backend_used="lfm2-prompt-router",
        model_id=ROUTER_MODEL_ID,
    )


class LFM2PromptRouterBackend(BaseRoutingBackend):
    name = "lfm2-prompt-router"
    aliases = ("lfm2.5-prompt-router", "prompt-router")
    model_id = ROUTER_MODEL_ID

    def load(self, *, device: str = "cpu") -> Any:
        return load_router_model(device)

    def route(
        self,
        text: str,
        routes: Sequence[str],
        *,
        device: str = "cpu",
        threshold: float | None = None,
    ) -> RoutingResult:
        cleaned = _validate_routes(routes)
        _validate_threshold(threshold)
        raw = _route_text(text, cleaned, device=device, threshold=threshold)
        return _routing_result(text, raw)

    def route_batch(
        self,
        texts: Sequence[str],
        routes: Sequence[str],
        *,
        device: str = "cpu",
        threshold: float | None = None,
    ) -> list[RoutingResult]:
        cleaned = _validate_routes(routes)
        _validate_threshold(threshold)
        # The remote code has no native batch entry point, so texts are
        # routed sequentially under the per-device inference lock.
        results: list[RoutingResult] = []
        for text in texts:
            raw = _route_text(text, cleaned, device=device, threshold=threshold)
            results.append(_routing_result(text, raw))
        return results


LFM2_PROMPT_ROUTER_BACKEND = LFM2PromptRouterBackend()

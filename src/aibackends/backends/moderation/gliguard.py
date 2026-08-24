from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any, cast

from aibackends.backends.moderation._base import BaseModerationBackend
from aibackends.core.exceptions import RuntimeImportError, TaskExecutionError
from aibackends.schemas.moderation import (
    HarmCategory,
    JailbreakStrategy,
    PromptModeration,
    RefusalVerdict,
    ResponseModeration,
    SafetyVerdict,
)

GLIGUARD_MODEL_ID = "fastino/gliguard-LLMGuardrails-300M"
DEFAULT_THRESHOLD = 0.5
DEFAULT_CATEGORY_THRESHOLD = 0.4
DEFAULT_BATCH_SIZE = 8

SAFETY_LABELS = ("safe", "unsafe")
REFUSAL_LABELS = ("refusal", "compliance")
TOXICITY_LABELS = (
    "violence_and_weapons",
    "non_violent_crime",
    "sexual_content",
    "hate_and_discrimination",
    "self_harm_and_suicide",
    "pii_exposure",
    "misinformation",
    "copyright_violation",
    "child_safety",
    "political_manipulation",
    "unethical_conduct",
    "regulated_advice",
    "privacy_violation",
    "other",
    "benign",
)
JAILBREAK_LABELS = (
    "prompt_injection",
    "jailbreak_attempt",
    "policy_evasion",
    "instruction_override",
    "system_prompt_exfiltration",
    "data_exfiltration",
    "roleplay_bypass",
    "hypothetical_bypass",
    "obfuscated_attack",
    "multi_step_attack",
    "social_engineering",
    "benign",
)

_MODEL_CACHE: dict[tuple[str, str], Any] = {}
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
        "Unsupported GliGuard device. Use 'cpu', 'gpu', 'cuda', 'cuda:<index>', or 'mps'."
    )


def load_gliguard_model(device: str = "cpu") -> Any:
    """Load GliGuard once per process and device."""
    device_name = normalize_device(device)
    cache_key = (GLIGUARD_MODEL_ID, device_name)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached
        try:
            from gliner2 import AutoExtractor
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[guardrails]' to use the GliGuard backend."
            ) from exc

        model = AutoExtractor.from_pretrained(
            GLIGUARD_MODEL_ID,
            map_location=device_name,
        )
        evaluate = getattr(model, "eval", None)
        if callable(evaluate):
            evaluate()
        _MODEL_CACHE[cache_key] = model
        _INFERENCE_LOCKS[cache_key] = threading.Lock()
        return model


def clear_model_cache() -> None:
    """Drop cached GliGuard models. Intended for tests and memory management."""
    with _CACHE_LOCK:
        _MODEL_CACHE.clear()
        _INFERENCE_LOCKS.clear()


def _prompt_schema(category_threshold: float) -> dict[str, Any]:
    return {
        "prompt_safety": list(SAFETY_LABELS),
        "prompt_toxicity": {
            "labels": list(TOXICITY_LABELS),
            "multi_label": True,
            "cls_threshold": category_threshold,
        },
        "jailbreak_detection": {
            "labels": list(JAILBREAK_LABELS),
            "multi_label": True,
            "cls_threshold": category_threshold,
        },
    }


def _response_schema(category_threshold: float) -> dict[str, Any]:
    return {
        "response_safety": list(SAFETY_LABELS),
        "response_toxicity": {
            "labels": list(TOXICITY_LABELS),
            "multi_label": True,
            "cls_threshold": category_threshold,
        },
        "response_refusal": list(REFUSAL_LABELS),
    }


def _validate_options(
    *,
    threshold: float,
    category_threshold: float,
    batch_size: int | None = None,
) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    if not 0.0 <= category_threshold <= 1.0:
        raise ValueError("category_threshold must be between 0 and 1.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1.")


def _inference_lock(device_name: str) -> threading.Lock:
    cache_key = (GLIGUARD_MODEL_ID, device_name)
    with _CACHE_LOCK:
        return _INFERENCE_LOCKS.setdefault(cache_key, threading.Lock())


def _classify_text(
    text: str,
    schema: dict[str, Any],
    *,
    device: str,
    threshold: float,
) -> dict[str, Any]:
    device_name = normalize_device(device)
    model = load_gliguard_model(device_name)
    with _inference_lock(device_name):
        result = model.classify_text(text, schema, threshold=threshold)
    if not isinstance(result, dict):
        raise TaskExecutionError("GliGuard returned an invalid classification result.")
    return cast(dict[str, Any], result)


def _batch_classify_text(
    texts: Sequence[str],
    schema: dict[str, Any],
    *,
    device: str,
    threshold: float,
    batch_size: int,
) -> list[dict[str, Any]]:
    device_name = normalize_device(device)
    model = load_gliguard_model(device_name)
    with _inference_lock(device_name):
        results = model.batch_classify_text(
            list(texts),
            schema,
            batch_size=batch_size,
            threshold=threshold,
        )
    if not isinstance(results, list) or len(results) != len(texts):
        raise TaskExecutionError("GliGuard returned an invalid number of batch results.")
    if any(not isinstance(result, dict) for result in results):
        raise TaskExecutionError("GliGuard returned an invalid batch classification result.")
    return cast(list[dict[str, Any]], results)


def _single_label(raw: dict[str, Any], task: str, allowed: tuple[str, ...]) -> str:
    value = raw.get(task)
    if not isinstance(value, str) or value not in allowed:
        raise TaskExecutionError(f"GliGuard returned an invalid {task!r} label: {value!r}.")
    return value


def _multi_labels(raw: dict[str, Any], task: str, allowed: tuple[str, ...]) -> list[str]:
    value = raw.get(task)
    if not isinstance(value, list) or any(
        not isinstance(label, str) or label not in allowed for label in value
    ):
        raise TaskExecutionError(f"GliGuard returned invalid {task!r} labels: {value!r}.")
    return cast(list[str], value)


def _prompt_result(prompt: str, raw: dict[str, Any]) -> PromptModeration:
    safety = cast(SafetyVerdict, _single_label(raw, "prompt_safety", SAFETY_LABELS))
    toxicity = cast(
        list[HarmCategory],
        _multi_labels(raw, "prompt_toxicity", TOXICITY_LABELS),
    )
    jailbreak = cast(
        list[JailbreakStrategy],
        _multi_labels(raw, "jailbreak_detection", JAILBREAK_LABELS),
    )
    return PromptModeration(
        prompt=prompt,
        is_safe=(
            safety == "safe"
            and all(label == "benign" for label in toxicity)
            and all(label == "benign" for label in jailbreak)
        ),
        safety=safety,
        toxicity=toxicity,
        jailbreak=jailbreak,
        backend_used="gliguard",
        model_id=GLIGUARD_MODEL_ID,
    )


def _response_result(
    response: str,
    prompt: str | None,
    raw: dict[str, Any],
) -> ResponseModeration:
    safety = cast(SafetyVerdict, _single_label(raw, "response_safety", SAFETY_LABELS))
    toxicity = cast(
        list[HarmCategory],
        _multi_labels(raw, "response_toxicity", TOXICITY_LABELS),
    )
    refusal = cast(
        RefusalVerdict,
        _single_label(raw, "response_refusal", REFUSAL_LABELS),
    )
    return ResponseModeration(
        response=response,
        prompt=prompt,
        is_safe=safety == "safe" and all(label == "benign" for label in toxicity),
        safety=safety,
        toxicity=toxicity,
        refusal=refusal,
        backend_used="gliguard",
        model_id=GLIGUARD_MODEL_ID,
    )


def _format_response(response: str, prompt: str | None) -> str:
    if prompt is None:
        return f"Response: {response}"
    return f"Prompt: {prompt}\nResponse: {response}"


class GliGuardBackend(BaseModerationBackend):
    name = "gliguard"
    aliases = ("gli-guard",)
    model_id = GLIGUARD_MODEL_ID

    def load(self, *, device: str = "cpu") -> Any:
        return load_gliguard_model(device)

    def moderate_prompt(
        self,
        prompt: str,
        *,
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
        category_threshold: float = DEFAULT_CATEGORY_THRESHOLD,
    ) -> PromptModeration:
        _validate_options(
            threshold=threshold,
            category_threshold=category_threshold,
        )
        raw = _classify_text(
            prompt,
            _prompt_schema(category_threshold),
            device=device,
            threshold=threshold,
        )
        return _prompt_result(prompt, raw)

    def moderate_response(
        self,
        response: str,
        *,
        prompt: str | None = None,
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
        category_threshold: float = DEFAULT_CATEGORY_THRESHOLD,
    ) -> ResponseModeration:
        _validate_options(
            threshold=threshold,
            category_threshold=category_threshold,
        )
        raw = _classify_text(
            _format_response(response, prompt),
            _response_schema(category_threshold),
            device=device,
            threshold=threshold,
        )
        return _response_result(response, prompt, raw)

    def moderate_prompts(
        self,
        prompts: Sequence[str],
        *,
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
        category_threshold: float = DEFAULT_CATEGORY_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[PromptModeration]:
        _validate_options(
            threshold=threshold,
            category_threshold=category_threshold,
            batch_size=batch_size,
        )
        if not prompts:
            return []
        raw_results = _batch_classify_text(
            prompts,
            _prompt_schema(category_threshold),
            device=device,
            threshold=threshold,
            batch_size=batch_size,
        )
        return [
            _prompt_result(prompt, raw)
            for prompt, raw in zip(prompts, raw_results, strict=True)
        ]

    def moderate_responses(
        self,
        responses: Sequence[str],
        *,
        prompts: Sequence[str | None] | None = None,
        device: str = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
        category_threshold: float = DEFAULT_CATEGORY_THRESHOLD,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[ResponseModeration]:
        _validate_options(
            threshold=threshold,
            category_threshold=category_threshold,
            batch_size=batch_size,
        )
        prompt_values = list(prompts) if prompts is not None else [None] * len(responses)
        if len(prompt_values) != len(responses):
            raise ValueError("prompts and responses must contain the same number of items.")
        if not responses:
            return []
        formatted = [
            _format_response(response, prompt)
            for response, prompt in zip(responses, prompt_values, strict=True)
        ]
        raw_results = _batch_classify_text(
            formatted,
            _response_schema(category_threshold),
            device=device,
            threshold=threshold,
            batch_size=batch_size,
        )
        return [
            _response_result(response, prompt, raw)
            for response, prompt, raw in zip(
                responses,
                prompt_values,
                raw_results,
                strict=True,
            )
        ]


GLIGUARD_BACKEND = GliGuardBackend()

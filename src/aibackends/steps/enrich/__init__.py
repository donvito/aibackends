from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from aibackends.core.config import (
    ensure_model_ref,
    ensure_runtime_spec,
    get_runtime_spec,
    parse_model_text,
)
from aibackends.core.registry import ModelRef, RuntimeSpec, TaskSpec
from aibackends.steps._base import BaseStep, StepContext
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import (
    build_messages,
    load_text_input,
    run_structured_task,
    run_text_task,
)
from aibackends.tasks.redact_pii import redact_pii
from aibackends.tasks.registry import create_task

IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def _coerce_payload(payload: Any) -> dict[str, Any]:
    return payload.copy() if isinstance(payload, dict) else {"input": payload}


def _resolve_content(data: dict[str, Any], payload: Any, input_key: str | None = None) -> Any:
    if input_key is not None and input_key in data:
        return data[input_key]
    return (
        data.get("transcript")
        or data.get("text")
        or data.get("brief")
        or data.get("audio_source")
        or data.get("path")
        or str(payload)
    )


def _extract_image_source(value: Any) -> Any | None:
    if isinstance(value, dict):
        if value.get("type") in {"image", "image_url"}:
            source = value.get("image", value.get("image_url"))
            if isinstance(source, dict):
                return source.get("url")
            return source
        return None
    if isinstance(value, Path):
        return value if value.suffix.lower() in IMAGE_SUFFIXES else None
    if not isinstance(value, str):
        return None
    if value.startswith("data:image/"):
        return value
    parsed = urlparse(value)
    if parsed.scheme in {"http", "https", "file"}:
        return value
    return value if Path(value).suffix.lower() in IMAGE_SUFFIXES else None


def _resolve_image_source(data: dict[str, Any], input_key: str | None = None) -> Any | None:
    candidate_keys = [input_key] if input_key is not None else []
    candidate_keys.extend(["image", "image_url", "path"])
    for key in candidate_keys:
        if key is None or key not in data:
            continue
        image_source = _extract_image_source(data[key])
        if image_source is not None:
            return image_source
    return None


def _resolve_vision_context_text(data: dict[str, Any]) -> str | None:
    for key in ("text", "transcript", "brief"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


class PIIRedactor(BaseStep):
    name = "pii_redact"

    def __init__(self, backend: str = "gliner", labels: list[str] | None = None) -> None:
        self.backend = backend
        self.labels = labels

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        source_text = data.get("transcript") or data.get("text") or data.get("brief")
        if source_text is None and data.get("path"):
            source_text = load_text_input(Path(data["path"]))
        redacted = redact_pii(source_text or "", backend=self.backend, labels=self.labels)
        if "transcript" in data:
            data["transcript"] = redacted.redacted_text
        else:
            data["text"] = redacted.redacted_text
        data["pii_redaction"] = redacted
        return data


class LLMTextGenerator(BaseStep):
    name = "llm_generate"

    def __init__(
        self,
        *,
        prompt: str,
        task_name: str = "generate",
        system_prompt: str = "You are a helpful AI assistant.",
        input_key: str | None = None,
        output_key: str | None = None,
        runtime: RuntimeSpec | None = None,
        model: ModelRef | None = None,
    ) -> None:
        runtime = ensure_runtime_spec(runtime)
        model = ensure_model_ref(model)
        self.prompt = prompt
        self.task_name = task_name
        self.system_prompt = system_prompt
        self.input_key = input_key
        self.output_key = output_key
        self.runtime = runtime
        self.model = model

    def run(self, payload: Any, context: StepContext) -> str | dict[str, Any]:
        data = _coerce_payload(payload)
        content = _resolve_content(data, payload, self.input_key)
        messages = build_messages(
            self.system_prompt,
            f"{self.prompt}\n\nContent:\n{content}",
        )
        result = run_text_task(
            task_name=self.task_name,
            messages=messages,
            runtime=self.runtime or context.runtime_config.runtime,
            model=self.model or context.runtime_config.model,
            **context.runtime_config.extra_options,
        )
        if self.output_key is None:
            return result
        data[self.output_key] = result
        return data


class TaskRunner(BaseStep):
    name = "task_run"

    def __init__(
        self,
        *,
        task: type[BaseTask] | TaskSpec,
        input_key: str | None = None,
        output_key: str | None = None,
        task_config: dict[str, Any] | None = None,
        run_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.task = task
        self.input_key = input_key
        self.output_key = output_key
        self.task_config = task_config or {}
        self.run_kwargs = run_kwargs or {}

    def run(self, payload: Any, context: StepContext) -> Any:
        data = _coerce_payload(payload)
        content = _resolve_content(data, payload, self.input_key)
        task_config = dict(context.runtime_config.extra_options)
        task_config.update(self.task_config)
        if isinstance(self.task, TaskSpec):
            if self.task.accepts_runtime and "runtime" not in task_config:
                task_config["runtime"] = (
                    get_runtime_spec(context.runtime_config.runtime)
                    if context.runtime_config.runtime
                    else None
                )
            if self.task.accepts_model and "model" not in task_config:
                task_config["model"] = parse_model_text(context.runtime_config.model)
        else:
            task_config.setdefault(
                "runtime",
                (
                    get_runtime_spec(context.runtime_config.runtime)
                    if context.runtime_config.runtime
                    else None
                ),
            )
            task_config.setdefault("model", parse_model_text(context.runtime_config.model))
        task_instance = create_task(self.task, **task_config)
        result = task_instance.run(content, **self.run_kwargs)
        if self.output_key is None:
            return result
        data[self.output_key] = result
        return data


class LLMAnalyser(BaseStep):
    name = "llm_analyse"

    def __init__(
        self,
        *,
        schema: type[BaseModel],
        prompt: str,
        task_name: str | None = None,
        system_prompt: str = "You are a structured analysis engine.",
        input_key: str | None = None,
        output_key: str | None = None,
        runtime: RuntimeSpec | None = None,
        model: ModelRef | None = None,
    ) -> None:
        runtime = ensure_runtime_spec(runtime)
        model = ensure_model_ref(model)
        self.schema = schema
        self.prompt = prompt
        self.task_name = task_name or prompt
        self.system_prompt = system_prompt
        self.input_key = input_key
        self.output_key = output_key
        self.runtime = runtime
        self.model = model

    def run(self, payload: Any, context: StepContext) -> BaseModel | dict[str, Any]:
        data = _coerce_payload(payload)
        content = _resolve_content(data, payload, self.input_key)
        messages = build_messages(
            self.system_prompt,
            f"{self.prompt}\n\nContent:\n{content}",
        )
        result = run_structured_task(
            task_name=self.task_name,
            schema=self.schema,
            messages=messages,
            runtime=self.runtime or context.runtime_config.runtime,
            model=self.model or context.runtime_config.model,
            **context.runtime_config.extra_options,
        )
        if self.output_key is None:
            return result
        data[self.output_key] = result
        return data


class VisionExtractor(LLMAnalyser):
    name = "vision_extract"

    def run(self, payload: Any, context: StepContext) -> BaseModel | dict[str, Any]:
        data = _coerce_payload(payload)
        image_source = _resolve_image_source(data, self.input_key)
        if image_source is None:
            return super().run(payload, context)

        prompt_text = self.prompt
        context_text = _resolve_vision_context_text(data)
        if context_text:
            prompt_text = f"{prompt_text}\n\nContext:\n{context_text}"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": image_source}},
                ],
            },
        ]
        result = run_structured_task(
            task_name=self.task_name,
            schema=self.schema,
            messages=messages,
            runtime=self.runtime or context.runtime_config.runtime,
            model=self.model or context.runtime_config.model,
            **context.runtime_config.extra_options,
        )
        if self.output_key is None:
            return result
        data[self.output_key] = result
        return data

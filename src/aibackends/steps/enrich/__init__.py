from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aibackends.steps._base import BaseStep, StepContext
from aibackends.tasks._utils import (
    build_messages,
    load_text_input,
    run_structured_task,
    run_text_task,
)
from aibackends.tasks.redact_pii import redact_pii
from aibackends.tasks.registry import create_task, get_task


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
        runtime: str | None = None,
        model: str | None = None,
    ) -> None:
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
        task_name: str,
        input_key: str | None = None,
        output_key: str | None = None,
        task_config: dict[str, Any] | None = None,
        run_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.task_name = task_name
        self.input_key = input_key
        self.output_key = output_key
        self.task_config = task_config or {}
        self.run_kwargs = run_kwargs or {}

    def run(self, payload: Any, context: StepContext) -> Any:
        data = _coerce_payload(payload)
        content = _resolve_content(data, payload, self.input_key)
        task_spec = get_task(self.task_name)
        task_config = dict(context.runtime_config.extra_options)
        task_config.update(self.task_config)
        if task_spec.accepts_runtime and "runtime" not in task_config:
            task_config["runtime"] = context.runtime_config.runtime
        if task_spec.accepts_model and "model" not in task_config:
            task_config["model"] = context.runtime_config.model
        task = create_task(self.task_name, **task_config)
        result = task.run(content, **self.run_kwargs)
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
        runtime: str | None = None,
        model: str | None = None,
    ) -> None:
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

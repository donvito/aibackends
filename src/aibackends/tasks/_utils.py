from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from aibackends.core.config import get_runtime, resolve_runtime_config
from aibackends.core.exceptions import TaskExecutionError, ValidationRetryExhaustedError
from aibackends.core.logging import emit_task_log
from aibackends.core.types import Message, TaskLog

T = TypeVar("T", bound=BaseModel)

TEXT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".log",
    ".md",
    ".rst",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def build_messages(system_prompt: str, user_prompt: str) -> list[Message]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def load_text_input(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.exists() or not path.is_file():
        return str(value)

    if path.suffix.lower() in TEXT_SUFFIXES:
        return path.read_text(encoding="utf-8", errors="ignore")

    if path.suffix.lower() == ".pdf":
        try:
            import fitz
        except ImportError as exc:
            raise TaskExecutionError(
                "Install 'aibackends[pdf]' to extract text from PDF inputs."
            ) from exc
        with fitz.open(path) as document:
            return "\n\n".join(page.get_text("text") for page in document)

    return (
        f"Input file: {path.name}\n"
        f"Absolute path: {path}\n"
        "Binary content was not inlined. Use a workflow step that can handle this file type."
    )


def parse_json_content(content: str) -> Any:
    text = content.strip()
    if not text:
        raise TaskExecutionError("Runtime returned empty content.")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in (r"(\{.*\})", r"(\[.*\])"):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

    raise TaskExecutionError("Could not locate valid JSON in runtime response.")


def run_structured_task(
    *,
    task_name: str,
    schema: type[T],
    messages: list[Message],
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> T:
    config = resolve_runtime_config({"runtime": runtime, "model": model, **overrides})
    runtime_client = get_runtime(config)
    started = time.perf_counter()
    emit_task_log(
        TaskLog(
            task_name=task_name,
            status="started",
            metadata={"runtime": config.runtime or "", "model": runtime_client.model_name},
        )
    )
    attempt_messages = list(messages)
    last_error: Exception | None = None
    for attempt in range(config.max_retries + 1):
        response = runtime_client.complete(attempt_messages, schema=schema)
        try:
            payload = parse_json_content(response.content)
            result = schema.model_validate(payload)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            emit_task_log(
                TaskLog(
                    task_name=task_name,
                    status="completed",
                    elapsed_ms=elapsed_ms,
                    metadata={
                        "runtime": config.runtime or "",
                        "model": runtime_client.model_name,
                        "tokens_in": response.usage.input_tokens or 0,
                        "tokens_out": response.usage.output_tokens or 0,
                    },
                )
            )
            return result
        except (ValidationError, TaskExecutionError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= config.max_retries:
                break
            attempt_messages.extend(
                [
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": (
                            f"Validation failed: {exc}. "
                            "Return valid JSON only and satisfy the schema exactly. "
                            "Do not return JSON Schema metadata such as title, type, "
                            "properties, items, anyOf, required, or $defs."
                        ),
                    },
                ]
            )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    emit_task_log(
        TaskLog(
            task_name=task_name,
            status="failed",
            elapsed_ms=elapsed_ms,
            metadata={"runtime": config.runtime or "", "error": str(last_error or "unknown")},
        )
    )
    raise ValidationRetryExhaustedError(
        f"{task_name} could not produce valid structured output "
        f"after {config.max_retries + 1} attempts."
    ) from last_error


def run_text_task(
    *,
    task_name: str,
    messages: list[Message],
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> str:
    config = resolve_runtime_config({"runtime": runtime, "model": model, **overrides})
    runtime_client = get_runtime(config)
    started = time.perf_counter()
    emit_task_log(
        TaskLog(
            task_name=task_name,
            status="started",
            metadata={"runtime": config.runtime or "", "model": runtime_client.model_name},
        )
    )
    response = runtime_client.complete(messages)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    emit_task_log(
        TaskLog(
            task_name=task_name,
            status="completed",
            elapsed_ms=elapsed_ms,
            metadata={
                "runtime": config.runtime or "",
                "model": runtime_client.model_name,
                "tokens_in": response.usage.input_tokens or 0,
                "tokens_out": response.usage.output_tokens or 0,
            },
        )
    )
    return response.content.strip()


def run_embedding_task(
    *,
    task_name: str,
    text: str,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> list[float]:
    config = resolve_runtime_config({"runtime": runtime, "model": model, **overrides})
    runtime_client = get_runtime(config)
    started = time.perf_counter()
    emit_task_log(
        TaskLog(
            task_name=task_name,
            status="started",
            metadata={"runtime": config.runtime or "", "model": runtime_client.model_name},
        )
    )
    vector = runtime_client.embed(text)
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    emit_task_log(
        TaskLog(
            task_name=task_name,
            status="completed",
            elapsed_ms=elapsed_ms,
            metadata={"runtime": config.runtime or "", "model": runtime_client.model_name},
        )
    )
    return vector

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

JSONDict = dict[str, Any]
Message = dict[str, Any]
StepCallback = Callable[["StepLog"], None]
PathLike = str | Path


class AIBackendsModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        populate_by_name=True,
        protected_namespaces=(),
    )


class TokenUsage(AIBackendsModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class RuntimeResponse(AIBackendsModel):
    content: str = ""
    model: str | None = None
    raw: JSONDict = Field(default_factory=dict)
    usage: TokenUsage = Field(default_factory=TokenUsage)


class StepLog(AIBackendsModel):
    task_name: str
    step_name: str
    status: Literal["started", "completed", "failed"]
    elapsed_ms: int | None = None
    metadata: JSONDict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TaskLog(AIBackendsModel):
    task_name: str
    status: Literal["started", "completed", "failed"]
    elapsed_ms: int | None = None
    metadata: JSONDict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BatchError(AIBackendsModel):
    input_path: str
    step_name: str
    exception: str
    retries_attempted: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class BatchRunResult(AIBackendsModel):
    results: list[Any] = Field(default_factory=list)
    errors: list[BatchError] = Field(default_factory=list)


class RuntimeConfig(AIBackendsModel):
    runtime: str | None = None
    model: str | None = None
    model_path: str | None = None
    adapter: str | None = None
    device: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    prompt_format: Literal["auto", "chat_template", "text"] = "auto"
    chat_template: str | None = None
    chat_template_path: str | None = None
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 2
    load_in_4bit: bool = False
    cache_dir: str | None = None
    on_step_complete: StepCallback | None = None
    extra_options: JSONDict = Field(default_factory=dict)

    @field_validator("cache_dir")
    @classmethod
    def expand_cache_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(Path(value).expanduser())

    @field_validator("chat_template_path")
    @classmethod
    def expand_chat_template_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return str(Path(value).expanduser())

    @model_validator(mode="after")
    def validate_chat_template_sources(self) -> "RuntimeConfig":
        if self.chat_template is not None and self.chat_template_path is not None:
            raise ValueError("Only one of `chat_template` or `chat_template_path` can be set.")
        return self

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aibackends.steps._base import BaseStep
from aibackends.tasks._utils import build_messages, load_text_input, run_structured_task
from aibackends.tasks.redact_pii import redact_pii


class PIIRedactor(BaseStep):
    name = "pii_redact"

    def __init__(self, backend: str = "gliner", labels: list[str] | None = None) -> None:
        self.backend = backend
        self.labels = labels

    def run(self, payload: Any, context: dict[str, Any]) -> dict[str, Any]:
        del context
        data = payload.copy() if isinstance(payload, dict) else {"input": payload}
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


class LLMAnalyser(BaseStep):
    name = "llm_analyse"

    def __init__(
        self,
        *,
        schema: type[BaseModel],
        prompt: str,
        runtime: str | None = None,
        model: str | None = None,
    ) -> None:
        self.schema = schema
        self.prompt = prompt
        self.runtime = runtime
        self.model = model

    def run(self, payload: Any, context: dict[str, Any]) -> BaseModel:
        del context
        data = payload.copy() if isinstance(payload, dict) else {"input": payload}
        content = (
            data.get("transcript")
            or data.get("text")
            or data.get("brief")
            or data.get("audio_source")
            or data.get("path")
            or str(payload)
        )
        messages = build_messages(
            "You are a structured analysis engine.",
            f"{self.prompt}\n\nContent:\n{content}",
        )
        return run_structured_task(
            task_name=self.prompt,
            schema=self.schema,
            messages=messages,
            runtime=self.runtime,
            model=self.model,
        )


class VisionExtractor(LLMAnalyser):
    name = "vision_extract"

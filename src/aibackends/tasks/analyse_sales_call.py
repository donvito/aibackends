from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.core.exceptions import TaskExecutionError
from aibackends.core.registry import TaskSpec
from aibackends.schemas.sales_call import SalesCallReport
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import build_messages, load_text_input, run_structured_task

AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".mp4", ".wav", ".webm"}


def analyse_sales_call(
    audio_or_transcript: str | Path,
    *,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> SalesCallReport:
    transcript = _load_transcript(audio_or_transcript)
    messages = build_messages(
        "You analyse B2B sales conversations and produce coaching-ready reports.",
        (
            "Analyse the sales call transcript. Estimate talk ratio for agent and customer, "
            "list objections, buying signals, action items, overall sentiment, "
            "and a score between 0 and 10.\n\n"
            f"Transcript:\n{transcript}"
        ),
    )
    return run_structured_task(
        task_name="analyse_sales_call",
        schema=SalesCallReport,
        messages=messages,
        runtime=runtime,
        model=model,
        **overrides,
    )


async def analyse_sales_call_async(
    audio_or_transcript: str | Path,
    *,
    runtime: str | None = None,
    model: str | None = None,
    **overrides: Any,
) -> SalesCallReport:
    return await asyncio.to_thread(
        analyse_sales_call,
        audio_or_transcript,
        runtime=runtime,
        model=model,
        **overrides,
    )


def _load_transcript(value: str | Path) -> str:
    path = Path(value).expanduser()
    if path.exists() and path.suffix.lower() in AUDIO_SUFFIXES:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TaskExecutionError(
                "Install 'aibackends[audio]' to transcribe audio files locally."
            ) from exc
        model_name = "base"
        model = WhisperModel(model_name, device="auto", compute_type="int8")
        segments, _info = model.transcribe(str(path))
        return "\n".join(segment.text.strip() for segment in segments if segment.text.strip())
    return load_text_input(value)


class AnalyseSalesCallTask(BaseTask):
    name = "analyse-sales-call"

    def run(
        self,
        input: str | Path,
        *,
        runtime: str | None = None,
        model: str | None = None,
        **overrides: Any,
    ) -> SalesCallReport:
        options = self._resolve_options(runtime=runtime, model=model, **overrides)
        return analyse_sales_call(input, **options)


TASK_SPEC = TaskSpec(
    name=AnalyseSalesCallTask.name,
    task_factory=AnalyseSalesCallTask,
    aliases=("analyse_sales_call",),
)

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aibackends.core.config import ensure_model_ref, ensure_runtime_spec
from aibackends.core.registry import ModelRef, RuntimeSpec, TaskSpec
from aibackends.schemas.video_ad import VideoAdReport
from aibackends.tasks._base import BaseTask
from aibackends.tasks._utils import build_messages, load_text_input, run_structured_task

VIDEO_SUFFIXES = {".avi", ".mov", ".mp4", ".mkv", ".webm"}


def analyse_video_ad(
    video_or_brief: str | Path,
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **overrides: Any,
) -> VideoAdReport:
    runtime = ensure_runtime_spec(runtime)
    model = ensure_model_ref(model)
    context = _load_video_context(video_or_brief)
    messages = build_messages(
        "You review short-form video ads for creative effectiveness.",
        (
            "Analyse this video ad or ad brief. "
            "Score the hook strength and CTA clarity from 0 to 10, "
            "summarize the key messages, and describe the emotional arc.\n\n"
            f"Context:\n{context}"
        ),
    )
    return run_structured_task(
        task_name="analyse_video_ad",
        schema=VideoAdReport,
        messages=messages,
        runtime=runtime,
        model=model,
        **overrides,
    )


async def analyse_video_ad_async(
    video_or_brief: str | Path,
    *,
    runtime: RuntimeSpec | None = None,
    model: ModelRef | None = None,
    **overrides: Any,
) -> VideoAdReport:
    return await asyncio.to_thread(
        analyse_video_ad,
        video_or_brief,
        runtime=runtime,
        model=model,
        **overrides,
    )


def _load_video_context(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.exists() or path.suffix.lower() not in VIDEO_SUFFIXES:
        return load_text_input(value)

    lines = [
        f"Video file: {path.name}",
        f"Absolute path: {path}",
        f"Size bytes: {path.stat().st_size}",
    ]
    try:
        import ffmpeg

        probe = ffmpeg.probe(str(path))
        format_data = probe.get("format", {})
        lines.append(f"Duration seconds: {format_data.get('duration', 'unknown')}")
        for stream in probe.get("streams", []):
            codec_type = stream.get("codec_type")
            if codec_type == "video":
                lines.append(
                    "Video stream: "
                    f"{stream.get('codec_name', 'unknown')} "
                    f"{stream.get('width', '?')}x{stream.get('height', '?')}"
                )
            if codec_type == "audio":
                lines.append(f"Audio stream: {stream.get('codec_name', 'unknown')}")
    except Exception:
        lines.append("ffmpeg probe unavailable; using file metadata only.")

    lines.append("Use the available metadata or surrounding description to assess the ad.")
    return "\n".join(lines)


class AnalyseVideoAdTask(BaseTask):
    name = "analyse-video-ad"

    def run(
        self,
        input: str | Path,
        *,
        runtime: RuntimeSpec | None = None,
        model: ModelRef | None = None,
        **overrides: Any,
    ) -> VideoAdReport:
        options = self._resolve_options(runtime=runtime, model=model, **overrides)
        return analyse_video_ad(input, **options)


TASK_SPEC = TaskSpec(
    name=AnalyseVideoAdTask.name,
    task_factory=AnalyseVideoAdTask,
    aliases=("analyse_video_ad",),
)

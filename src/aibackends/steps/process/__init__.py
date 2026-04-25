from __future__ import annotations

from pathlib import Path
from typing import Any

from aibackends.core.exceptions import TaskExecutionError
from aibackends.steps._base import BaseStep


class ImageRenderer(BaseStep):
    name = "image_render"

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi

    def run(self, payload: Any, context: dict[str, Any]) -> dict[str, Any]:
        del context
        data = payload.copy() if isinstance(payload, dict) else {"input": payload}
        path = Path(data["path"]).expanduser()
        data["render_dpi"] = self.dpi
        try:
            import fitz
        except ImportError:
            data["page_count"] = 1
            return data

        with fitz.open(path) as document:
            data["page_count"] = len(document)
        return data


class WhisperTranscriber(BaseStep):
    name = "whisper_transcribe"

    def __init__(self, model_name: str = "base") -> None:
        self.model_name = model_name

    def run(self, payload: Any, context: dict[str, Any]) -> dict[str, Any]:
        del context
        data = payload.copy() if isinstance(payload, dict) else {"input": payload}
        if data.get("transcript"):
            return data
        path = Path(data["path"]).expanduser()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TaskExecutionError(
                "Install 'aibackends[audio]' to enable Whisper transcription steps."
            ) from exc
        model = WhisperModel(self.model_name, device="auto", compute_type="int8")
        segments, _info = model.transcribe(str(path))
        data["transcript"] = "\n".join(
            segment.text.strip() for segment in segments if segment.text.strip()
        )
        return data


class FrameExtractor(BaseStep):
    name = "frame_extract"

    def __init__(self, sample_every_seconds: int = 5) -> None:
        self.sample_every_seconds = sample_every_seconds

    def run(self, payload: Any, context: dict[str, Any]) -> dict[str, Any]:
        del context
        data = payload.copy() if isinstance(payload, dict) else {"input": payload}
        data["frame_sampling_seconds"] = self.sample_every_seconds
        return data


class AudioStripper(BaseStep):
    name = "audio_strip"

    def run(self, payload: Any, context: dict[str, Any]) -> dict[str, Any]:
        del context
        data = payload.copy() if isinstance(payload, dict) else {"input": payload}
        data["audio_source"] = data.get("path")
        return data

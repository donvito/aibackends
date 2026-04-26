from __future__ import annotations

from pathlib import Path
from typing import Any

from aibackends.steps._base import BaseStep, StepContext
from aibackends.tasks._utils import TEXT_SUFFIXES, load_text_input

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


def _resolve_source(data: dict[str, Any]) -> Path:
    source = data.get("input", data.get("path"))
    if source is None:
        raise ValueError("Workflow ingest step requires an input or path value.")
    return Path(str(source)).expanduser()


class FileIngestor(BaseStep):
    name = "file_ingest"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        source = _resolve_source(data)
        data["path"] = str(source)
        if source.exists() and source.suffix.lower() in TEXT_SUFFIXES | {".pdf"}:
            data["text"] = load_text_input(source)
        return data


class PDFIngestor(FileIngestor):
    name = "pdf_ingest"


class ImageIngestor(BaseStep):
    name = "image_ingest"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        source = _resolve_source(data)
        data["path"] = str(source)
        if source.suffix.lower() in IMAGE_SUFFIXES:
            data["image"] = str(source)
        return data


class AudioIngestor(BaseStep):
    name = "audio_ingest"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        source = _resolve_source(data)
        data["path"] = str(source)
        if source.exists() and source.suffix.lower() in TEXT_SUFFIXES:
            data["transcript"] = load_text_input(source)
        return data


class VideoIngestor(BaseStep):
    name = "video_ingest"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = _coerce_payload(payload)
        source = _resolve_source(data)
        data["path"] = str(source)
        if source.exists() and source.suffix.lower() in TEXT_SUFFIXES:
            data["brief"] = load_text_input(source)
        return data

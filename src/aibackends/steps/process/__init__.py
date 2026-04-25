from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from aibackends.core.config import get_runtime
from aibackends.core.exceptions import TaskExecutionError
from aibackends.schemas.embeddings import (
    EmbeddedText,
    EmbeddingSimilarityResult,
    SimilarityPair,
)
from aibackends.steps._base import BaseStep, StepContext


def _coerce_texts(payload: Any) -> list[str]:
    raw_texts = payload.get("texts") if isinstance(payload, dict) else payload
    if not isinstance(raw_texts, list):
        raise TaskExecutionError(
            "Embedding similarity expects a list of texts or a dict with a 'texts' list."
        )
    if len(raw_texts) < 2:
        raise TaskExecutionError("Embedding similarity requires at least two texts.")

    texts: list[str] = []
    for index, value in enumerate(raw_texts):
        if not isinstance(value, str):
            raise TaskExecutionError(
                f"Embedding similarity text at index {index} must be a string."
            )
        if not value.strip():
            raise TaskExecutionError(
                f"Embedding similarity text at index {index} must not be empty."
            )
        texts.append(value)
    return texts


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise TaskExecutionError("Embedding similarity requires embeddings with matching sizes.")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    score = dot_product / (left_norm * right_norm)
    return max(-1.0, min(1.0, score))


class ImageRenderer(BaseStep):
    name = "image_render"

    def __init__(self, dpi: int = 150) -> None:
        self.dpi = dpi

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
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

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
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

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = payload.copy() if isinstance(payload, dict) else {"input": payload}
        data["frame_sampling_seconds"] = self.sample_every_seconds
        return data


class AudioStripper(BaseStep):
    name = "audio_strip"

    def run(self, payload: Any, context: StepContext) -> dict[str, Any]:
        del context
        data = payload.copy() if isinstance(payload, dict) else {"input": payload}
        data["audio_source"] = data.get("path")
        return data


class EmbeddingSimilarityComparer(BaseStep):
    name = "embedding_similarity"

    def run(self, payload: Any, context: StepContext) -> EmbeddingSimilarityResult:
        texts = _coerce_texts(payload)
        runtime_client = get_runtime(context.runtime_config)
        embeddings = [runtime_client.embed(text) for text in texts]
        embedded_texts = [
            EmbeddedText(
                index=index,
                text=text,
                embedding=embedding,
                dimension=len(embedding),
            )
            for index, (text, embedding) in enumerate(zip(texts, embeddings, strict=True))
        ]

        similarity_matrix: list[list[float]] = []
        ranked_pairs: list[SimilarityPair] = []
        for left_index, left in enumerate(embedded_texts):
            row: list[float] = []
            for right_index, right in enumerate(embedded_texts):
                score = _cosine_similarity(left.embedding, right.embedding)
                row.append(score)
                if right_index > left_index:
                    ranked_pairs.append(
                        SimilarityPair(
                            left_index=left_index,
                            right_index=right_index,
                            left_text=left.text,
                            right_text=right.text,
                            cosine_similarity=score,
                        )
                    )
            similarity_matrix.append(row)

        ranked_pairs.sort(
            key=lambda pair: (-pair.cosine_similarity, pair.left_index, pair.right_index)
        )
        return EmbeddingSimilarityResult(
            texts=texts,
            embeddings=embedded_texts,
            similarity_matrix=similarity_matrix,
            ranked_pairs=ranked_pairs,
        )

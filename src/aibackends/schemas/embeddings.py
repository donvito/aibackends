from __future__ import annotations

from aibackends.core.types import AIBackendsModel


class EmbeddedText(AIBackendsModel):
    index: int
    text: str
    embedding: list[float]
    dimension: int


class SimilarityPair(AIBackendsModel):
    left_index: int
    right_index: int
    left_text: str
    right_text: str
    cosine_similarity: float


class EmbeddingSimilarityResult(AIBackendsModel):
    texts: list[str]
    embeddings: list[EmbeddedText]
    similarity_matrix: list[list[float]]
    ranked_pairs: list[SimilarityPair]

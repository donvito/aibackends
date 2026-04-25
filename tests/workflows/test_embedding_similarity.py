from __future__ import annotations

import math

import pytest

from aibackends.core.exceptions import WorkflowStepError
from aibackends.core.registry import ModelRef
from aibackends.runtimes import get_runtime_spec
from aibackends.workflows import EmbeddingSimilarityWorkflow, create_workflow, get_workflow


def _stub_score(left_length: int, right_length: int) -> float:
    left = [float(left_length), 1.0, 0.5]
    right = [float(right_length), 1.0, 0.5]
    dot_product = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_norm * right_norm)


def test_embedding_similarity_workflow_returns_ranked_cosine_scores():
    texts = ["cat", "bear", "elephant"]
    workflow = EmbeddingSimilarityWorkflow(
        runtime=get_runtime_spec("stub"),
        model=ModelRef(name="stub-model"),
    )

    result = workflow.run(texts)

    assert result.texts == texts
    assert [item.dimension for item in result.embeddings] == [3, 3, 3]
    assert result.similarity_matrix[0][0] == pytest.approx(1.0)
    assert result.similarity_matrix[0][1] == pytest.approx(_stub_score(3, 4))
    assert result.similarity_matrix[1][0] == pytest.approx(_stub_score(3, 4))
    assert [(pair.left_index, pair.right_index) for pair in result.ranked_pairs] == [
        (0, 1),
        (1, 2),
        (0, 2),
    ]
    assert result.ranked_pairs[0].cosine_similarity > result.ranked_pairs[-1].cosine_similarity


def test_embedding_similarity_workflow_accepts_dict_payload():
    workflow = EmbeddingSimilarityWorkflow(
        runtime=get_runtime_spec("stub"),
        model=ModelRef(name="stub-model"),
    )

    result = workflow.run({"texts": ["cat", "bear"]})

    assert [item.text for item in result.embeddings] == ["cat", "bear"]
    assert len(result.ranked_pairs) == 1


def test_embedding_similarity_workflow_requires_two_texts():
    workflow = EmbeddingSimilarityWorkflow(
        runtime=get_runtime_spec("stub"),
        model=ModelRef(name="stub-model"),
    )

    with pytest.raises(WorkflowStepError, match="at least two texts"):
        workflow.run(["solo"])


def test_embedding_similarity_workflow_is_available_via_registry():
    workflow = create_workflow(
        get_workflow("embedding_similarity"),
        runtime=get_runtime_spec("stub"),
        model=ModelRef(name="stub-model"),
    )

    assert isinstance(workflow, EmbeddingSimilarityWorkflow)

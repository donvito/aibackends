from aibackends.core.registry import WorkflowSpec
from aibackends.steps.process import EmbeddingSimilarityComparer
from aibackends.workflows._base import Pipeline


class EmbeddingSimilarityWorkflow(Pipeline):
    steps = [EmbeddingSimilarityComparer()]


WORKFLOW_SPEC = WorkflowSpec(
    name="embedding-similarity",
    workflow_factory=EmbeddingSimilarityWorkflow,
    aliases=(
        "embedding_similarity",
        "embedding-similarity-workflow",
        "embedding_similarity_workflow",
    ),
)

from aibackends.core.registry import WorkflowSpec
from aibackends.steps.enrich import PIIRedactor as PIIRedactorStep
from aibackends.steps.ingest import FileIngestor
from aibackends.steps.output import OutputPassthrough
from aibackends.workflows._base import Pipeline


class PIIRedactorWorkflow(Pipeline):
    def __init__(
        self,
        *,
        pii_backend: str = "gliner",
        labels: list[str] | None = None,
        runtime: str | None = None,
        model: str | None = None,
        **overrides,
    ) -> None:
        self.steps = [
            FileIngestor(),
            PIIRedactorStep(backend=pii_backend, labels=labels),
            OutputPassthrough(),
        ]
        super().__init__(runtime=runtime, model=model, **overrides)


PIIRedactor = PIIRedactorWorkflow

WORKFLOW_SPEC = WorkflowSpec(
    name="pii-redactor",
    workflow_factory=PIIRedactorWorkflow,
    aliases=("pii_redactor", "pii-redactor-workflow", "pii_redactor_workflow"),
)

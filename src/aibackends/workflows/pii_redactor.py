from aibackends.steps.enrich import PIIRedactor as PIIRedactorStep
from aibackends.steps.ingest import FileIngestor
from aibackends.steps.output import OutputPassthrough
from aibackends.workflows._base import Pipeline


class PIIRedactorWorkflow(Pipeline):
    steps = [
        FileIngestor(),
        PIIRedactorStep(backend="gliner"),
        OutputPassthrough(),
    ]


PIIRedactor = PIIRedactorWorkflow

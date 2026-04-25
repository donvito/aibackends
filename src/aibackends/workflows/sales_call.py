from aibackends.schemas.sales_call import SalesCallReport
from aibackends.steps.enrich import LLMAnalyser, PIIRedactor
from aibackends.steps.ingest import AudioIngestor
from aibackends.steps.process import WhisperTranscriber
from aibackends.steps.validate import PydanticValidator
from aibackends.workflows._base import Pipeline


class SalesCallAnalyser(Pipeline):
    steps = [
        AudioIngestor(),
        WhisperTranscriber(),
        PIIRedactor(backend="gliner"),
        LLMAnalyser(schema=SalesCallReport, prompt="Analyse the sales call transcript."),
        PydanticValidator(schema=SalesCallReport),
    ]

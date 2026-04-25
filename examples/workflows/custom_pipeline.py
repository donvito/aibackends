from pathlib import Path

from pydantic import BaseModel

from aibackends.steps.enrich import LLMAnalyser
from aibackends.steps.ingest import FileIngestor
from aibackends.steps.validate import PydanticValidator
from aibackends.workflows import Pipeline


class LeadBrief(BaseModel):
    name: str
    company: str | None = None
    email: str | None = None
    priority: str | None = None
    next_step: str | None = None


class LeadIntakePipeline(Pipeline):
    steps = [
        FileIngestor(),
        LLMAnalyser(schema=LeadBrief, prompt="Extract the lead details from the note."),
        PydanticValidator(schema=LeadBrief),
    ]


lead_note_path = Path(__file__).parent.parent / "data" / "lead_note.txt"
workflow = LeadIntakePipeline(runtime="llamacpp", model="gemma4-e2b")
result = workflow.run(lead_note_path)
print(result.model_dump_json(indent=2))

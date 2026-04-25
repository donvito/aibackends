from pathlib import Path

from pydantic import BaseModel

from aibackends import configure
from aibackends.tasks import extract


class Lead(BaseModel):
    name: str
    company: str | None = None
    email: str | None = None
    estimated_budget: float | None = None
    priority: str | None = None
    next_step: str | None = None


configure(runtime="llamacpp", model="gemma4-e2b")

lead_note_path = Path(__file__).parent.parent / "data" / "lead_note.txt"
result = extract(
    lead_note_path,
    schema=Lead,
    instructions="Extract the lead details from the intake note.",
)

print(result.model_dump_json(indent=2))

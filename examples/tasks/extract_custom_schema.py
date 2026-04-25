from pathlib import Path

from pydantic import BaseModel

from aibackends.tasks import create_task


class Lead(BaseModel):
    name: str
    company: str | None = None
    email: str | None = None
    estimated_budget: float | None = None
    priority: str | None = None
    next_step: str | None = None


task = create_task(
    "extract",
    runtime="llamacpp",
    model="gemma4-e2b",
    schema=Lead,
    instructions="Extract the lead details from the intake note.",
)

lead_note_path = Path(__file__).parent.parent / "data" / "lead_note.txt"
result = task.run(lead_note_path)

print(result.model_dump_json(indent=2))

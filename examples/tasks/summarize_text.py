from pathlib import Path

from aibackends.tasks import create_task

task = create_task(
    "summarize",
    runtime="llamacpp",
    model="gemma4-e2b",
)

notes_path = Path(__file__).parent.parent / "data" / "meeting_notes.txt"
summary = task.run(notes_path)

print(summary)

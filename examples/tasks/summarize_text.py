from pathlib import Path

from aibackends import configure
from aibackends.tasks import summarize

configure(runtime="llamacpp", model="gemma4-e2b")

notes_path = Path(__file__).parent.parent / "data" / "meeting_notes.txt"
summary = summarize(notes_path)

print(summary)

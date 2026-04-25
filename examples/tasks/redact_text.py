from pathlib import Path

from aibackends.tasks import create_task

task = create_task(
    "redact-pii",
    backend="gliner",
    labels=["email", "phone_number", "address"],
)

note_path = Path(__file__).parent.parent / "data" / "contract.txt"
result = task.run(note_path)

print(result.model_dump_json(indent=2))

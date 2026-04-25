from pathlib import Path

from aibackends.tasks import create_task

task = create_task(
    "redact-pii",
    backend="gliner",
    labels=["name",
            "email",
            "phone_number",
            "address",
            "idenfication_number",
            "passport_number",
            "account_number"],
)

note_path = Path(__file__).parent.parent / "data" / "contract.txt"
result = task.run(note_path)

print(result.model_dump_json(indent=2))

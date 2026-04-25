from pathlib import Path

from aibackends.tasks import redact_pii

note_path = Path(__file__).parent.parent / "data" / "contract.txt"
result = redact_pii(note_path, backend="gliner", labels=["email", "phone_number", "address"])

print(result.model_dump_json(indent=2))

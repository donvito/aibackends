from pathlib import Path

from aibackends.tasks import create_task

task = create_task(
    "classify",
    runtime="llamacpp",
    model="gemma4-e2b",
    labels=["invoice", "rental contract", "employment contract", "receipt", "sales_call"],
)

document_path = Path(__file__).parent.parent / "data" / "contract.txt"
classification = task.run(document_path)

print(classification.model_dump_json(indent=2))

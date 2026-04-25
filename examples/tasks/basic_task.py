from pathlib import Path

from aibackends.tasks import create_task

task = create_task(
    "extract-invoice",
    runtime="llamacpp",
    model="gemma4-e2b",
)

invoice_path = Path(__file__).parent.parent / "data" / "invoice.txt"
result = task.run(invoice_path)
print(result.model_dump_json(indent=2))

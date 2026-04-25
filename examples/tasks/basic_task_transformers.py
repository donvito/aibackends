from pathlib import Path

from aibackends.tasks import create_task

# Use the small instruction-tuned Gemma 3 variant so the
# Transformers example behaves like a chat model on CPU.
task = create_task(
    "extract-invoice",
    runtime="transformers",
    model="google/gemma-3-270m-it",
    max_tokens=256,
)


invoice_path = Path(__file__).parent.parent / "data" / "invoice.txt"
result = task.run(invoice_path)
print(result.model_dump_json(indent=2))

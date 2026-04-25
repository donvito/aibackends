from pathlib import Path

from aibackends import configure
from aibackends.tasks import extract_invoice

# Use the small instruction-tuned Gemma 3 variant so the
# Transformers example behaves like a chat model on CPU.
configure(runtime="transformers", model="google/gemma-3-270m-it")


invoice_path = Path(__file__).parent.parent / "data" / "invoice.txt"
result = extract_invoice(invoice_path, max_tokens=256)
print(result.model_dump_json(indent=2))

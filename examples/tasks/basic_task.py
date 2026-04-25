from pathlib import Path

from aibackends import configure
from aibackends.tasks import extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")

invoice_path = Path(__file__).parent.parent / "data" / "invoice.txt"
result = extract_invoice(invoice_path)
print(result.model_dump_json(indent=2))

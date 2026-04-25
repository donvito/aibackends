from pathlib import Path

from aibackends import configure
from aibackends.tasks import classify

configure(runtime="llamacpp", model="gemma4-e2b")

document_path = Path(__file__).parent.parent / "data" / "contract.txt"
classification = classify(
    document_path,
    labels=["invoice", "rental contract", "employment contract", "receipt", "sales_call"],
)

print(classification.model_dump_json(indent=2))

from pathlib import Path

from aibackends import configure
from aibackends.tasks import analyse_sales_call

configure(runtime="llamacpp", model="gemma4-e2b")

transcript_path = Path(__file__).parent.parent / "data" / "batch" / "sales_call_1.txt"
report = analyse_sales_call(transcript_path)

print(report.model_dump_json(indent=2))

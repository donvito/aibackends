from pathlib import Path

from aibackends.tasks import create_task

task = create_task(
    "analyse-sales-call",
    runtime="llamacpp",
    model="gemma4-e2b",
)

transcript_path = Path(__file__).parent.parent / "data" / "batch" / "sales_call_1.txt"
report = task.run(transcript_path)

print(report.model_dump_json(indent=2))

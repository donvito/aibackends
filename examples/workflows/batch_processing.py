from pathlib import Path

from aibackends.workflows import create_workflow

data_dir = Path(__file__).parent.parent / "data" / "batch"
transcript_paths = sorted(data_dir.glob("sales_call_*.txt"))

print(f"Found {len(transcript_paths)} transcripts in {data_dir}", flush=True)
for path in transcript_paths:
    print(f"- {path.name}", flush=True)

workflow = create_workflow(
    "sales-call",
    runtime="llamacpp",
    model="gemma4-e2b",
)

print("Running sales-call workflow batch...", flush=True)
results = workflow.run_batch(
    inputs=transcript_paths,
    max_concurrency=4,
    on_error="collect",
)

print(f"Completed: {len(results.results)} results, {len(results.errors)} errors", flush=True)
print(results.model_dump_json(indent=2))

from pathlib import Path

from aibackends import configure
from aibackends.workflows import SalesCallAnalyser

configure(runtime="llamacpp", model="gemma4-e2b")

transcript_paths = sorted((Path(__file__).parent.parent / "data" / "batch").glob("sales_call_*.txt"))
results = SalesCallAnalyser().run_batch(
    inputs=transcript_paths,
    max_concurrency=4,
    on_error="collect",
)

print(results.model_dump_json(indent=2))

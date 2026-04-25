import sys
from pathlib import Path

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.workflows import SalesCallAnalyser, create_workflow


def main() -> None:
    try:
        data_dir = Path(__file__).parent.parent / "data" / "batch"
        transcript_paths = sorted(data_dir.glob("sales_call_*.txt"))

        print(f"Found {len(transcript_paths)} transcripts in {data_dir}", flush=True)
        for path in transcript_paths:
            print(f"- {path.name}", flush=True)

            workflow = create_workflow(
                SalesCallAnalyser,
                runtime=LLAMACPP,
                model=GEMMA4_E2B,
            )

        print("Running sales-call workflow batch...", flush=True)
        results = workflow.run_batch(
            inputs=transcript_paths,
            max_concurrency=4,
            on_error="collect",
        )

        print(
            f"Completed: {len(results.results)} results, {len(results.errors)} errors",
            flush=True,
        )
        print(results.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

import sys
from pathlib import Path

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import AnalyseSalesCallTask, create_task


def main() -> None:
    try:
        task = create_task(
            AnalyseSalesCallTask,
            runtime=LLAMACPP,
            model=GEMMA4_E2B,
        )
        transcript_path = Path(__file__).parent.parent / "data" / "batch" / "sales_call_1.txt"
        report = task.run(transcript_path)
        print(report.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

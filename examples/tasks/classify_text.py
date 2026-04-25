import sys
from pathlib import Path

from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import create_task


def main() -> None:
    try:
        task = create_task(
            "classify",
            runtime="llamacpp",
            model="gemma4-e2b",
            labels=["invoice", "rental contract", "employment contract", "receipt", "sales_call"],
        )
        document_path = Path(__file__).parent.parent / "data" / "contract.txt"
        classification = task.run(document_path)
        print(classification.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

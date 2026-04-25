import sys
from pathlib import Path

from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import create_task


def main() -> None:
    try:
        task = create_task(
            "redact-pii",
            backend="gliner",
            labels=[
                "name",
                "email",
                "phone_number",
                "address",
                "idenfication_number",
                "passport_number",
                "account_number",
            ],
        )
        note_path = Path(__file__).parent.parent / "data" / "contract.txt"
        result = task.run(note_path)
        print(result.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

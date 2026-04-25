import sys
from pathlib import Path

from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import create_task

# Use the small instruction-tuned Gemma 3 variant so the
# Transformers example behaves like a chat model on CPU.

def main() -> None:
    try:
        task = create_task(
            "extract-invoice",
            runtime="transformers",
            model="google/gemma-3-270m-it",
            max_tokens=256,
        )
        invoice_path = Path(__file__).parent.parent / "data" / "invoice.txt"
        result = task.run(invoice_path)
        print(result.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

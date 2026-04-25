import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import ExtractInvoiceTask, create_task


def main() -> None:
    try:
        from crewai.tools import tool
    except ImportError as exc:
        print(
            "This example requires the 'crewai' package. Install it and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        invoice_task = create_task(ExtractInvoiceTask, runtime=LLAMACPP, model=GEMMA4_E2B)

        @tool("Invoice Extractor")
        def process_invoice(file_path: str) -> str:
            return invoice_task.run(file_path).model_dump_json()

        _ = process_invoice
        print("Configured CrewAI tool `Invoice Extractor`.", flush=True)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

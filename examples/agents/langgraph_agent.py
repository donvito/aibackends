import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.tasks import ExtractInvoiceTask, RedactPIITask, create_task


def main() -> None:
    try:
        from langchain_core.tools import tool
        from langgraph.prebuilt import create_react_agent
    except ImportError as exc:
        print(
            "This example requires the 'langgraph' and 'langchain-core' packages. "
            "Install them and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        invoice_task = create_task(ExtractInvoiceTask, runtime=LLAMACPP, model=GEMMA4_E2B)
        redaction_task = create_task(RedactPIITask, backend="gliner")

        @tool
        def process_invoice(file_path: str) -> dict:
            """Extract structured data from an invoice PDF."""
            return invoice_task.run(file_path).model_dump()

        @tool
        def redact_text(text: str) -> str:
            """Remove PII from text."""
            return redaction_task.run(text).redacted_text

        create_react_agent(model="openai:gpt-4o-mini", tools=[process_invoice, redact_text])
        print("Configured LangGraph agent with invoice and redaction tools.", flush=True)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

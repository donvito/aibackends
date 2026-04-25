import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import create_task


def main() -> None:
    try:
        from agents import Agent, function_tool
    except ImportError as exc:
        print(
            "This example requires the 'openai-agents' package. Install it and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")

        @function_tool
        def process_invoice(file_path: str) -> str:
            """Extract structured data from an invoice PDF."""
            return invoice_task.run(file_path).model_dump_json()

        Agent(name="doc-processor", tools=[process_invoice])
        print("Configured OpenAI Agents SDK agent `doc-processor`.", flush=True)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import create_task


def main() -> None:
    try:
        from pydantic_ai import Agent
    except ImportError as exc:
        print(
            "This example requires the 'pydantic-ai' package. Install it and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")
        classify_task = create_task(
            "classify",
            runtime="llamacpp",
            model="gemma4-e2b",
            labels=["invoice", "contract", "receipt"],
        )

        def extract_invoice(file_path: str):
            return invoice_task.run(file_path)

        def classify(text: str):
            return classify_task.run(text)

        agent = Agent(model="openai:gpt-4o-mini", tools=[extract_invoice, classify])
        result = agent.run_sync("Extract the data from invoice.pdf and classify the document type.")
        print(result.output)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import create_task


def main() -> None:
    try:
        from agno.agent import Agent
        from agno.tools import tool
    except ImportError as exc:
        print(
            "This example requires the 'agno' package. Install it and try again.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")

        @tool
        def process_invoice(file_path: str) -> dict:
            return invoice_task.run(file_path).model_dump()

        Agent(tools=[process_invoice])
        print("Configured Agno agent with `process_invoice`.", flush=True)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

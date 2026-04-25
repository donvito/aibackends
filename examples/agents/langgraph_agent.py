from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from aibackends.tasks import create_task

invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")
redaction_task = create_task("redact-pii", backend="gliner")


@tool
def process_invoice(file_path: str) -> dict:
    """Extract structured data from an invoice PDF."""
    return invoice_task.run(file_path).model_dump()


@tool
def redact_text(text: str) -> str:
    """Remove PII from text."""
    return redaction_task.run(text).redacted_text


agent = create_react_agent(model="openai:gpt-4o-mini", tools=[process_invoice, redact_text])

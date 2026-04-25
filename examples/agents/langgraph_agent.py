from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from aibackends import configure
from aibackends.tasks import extract_invoice, redact_pii

configure(runtime="llamacpp", model="gemma4-e2b")


@tool
def process_invoice(file_path: str) -> dict:
    """Extract structured data from an invoice PDF."""
    return extract_invoice(file_path).model_dump()


@tool
def redact_text(text: str) -> str:
    """Remove PII from text."""
    return redact_pii(text, backend="gliner").redacted_text


agent = create_react_agent(model="openai:gpt-4o-mini", tools=[process_invoice, redact_text])

from agents import Agent, function_tool

from aibackends import configure
from aibackends.tasks import extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")


@function_tool
def process_invoice(file_path: str) -> str:
    """Extract structured data from an invoice PDF."""
    return extract_invoice(file_path).model_dump_json()


agent = Agent(name="doc-processor", tools=[process_invoice])

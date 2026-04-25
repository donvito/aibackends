from agents import Agent, function_tool

from aibackends.tasks import create_task

invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")


@function_tool
def process_invoice(file_path: str) -> str:
    """Extract structured data from an invoice PDF."""
    return invoice_task.run(file_path).model_dump_json()


agent = Agent(name="doc-processor", tools=[process_invoice])

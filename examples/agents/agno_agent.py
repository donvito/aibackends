from agno.agent import Agent
from agno.tools import tool

from aibackends.tasks import create_task

invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")


@tool
def process_invoice(file_path: str) -> dict:
    return invoice_task.run(file_path).model_dump()


agent = Agent(tools=[process_invoice])

from agno.agent import Agent
from agno.tools import tool

from aibackends import configure
from aibackends.tasks import extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")


@tool
def process_invoice(file_path: str) -> dict:
    return extract_invoice(file_path).model_dump()


agent = Agent(tools=[process_invoice])

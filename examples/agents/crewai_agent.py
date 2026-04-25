from crewai.tools import tool

from aibackends import configure
from aibackends.tasks import extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")


@tool("Invoice Extractor")
def process_invoice(file_path: str) -> str:
    return extract_invoice(file_path).model_dump_json()

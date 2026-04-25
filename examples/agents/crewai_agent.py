from crewai.tools import tool

from aibackends.tasks import create_task

invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")


@tool("Invoice Extractor")
def process_invoice(file_path: str) -> str:
    return invoice_task.run(file_path).model_dump_json()

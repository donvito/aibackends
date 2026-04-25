from llama_index.core.tools import FunctionTool

from aibackends.tasks import create_task

invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")


def extract_invoice(file_path: str):
    return invoice_task.run(file_path)


invoice_tool = FunctionTool.from_defaults(fn=extract_invoice)

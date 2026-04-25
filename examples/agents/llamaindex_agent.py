from llama_index.core.tools import FunctionTool

from aibackends import configure
from aibackends.tasks import extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")

invoice_tool = FunctionTool.from_defaults(fn=extract_invoice)

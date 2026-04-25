from pydantic_ai import Agent

from aibackends.tasks import create_task

invoice_task = create_task("extract-invoice", runtime="llamacpp", model="gemma4-e2b")
classify_task = create_task(
    "classify",
    runtime="llamacpp",
    model="gemma4-e2b",
    labels=["invoice", "contract", "receipt"],
)


def extract_invoice(file_path: str):
    return invoice_task.run(file_path)


def classify(text: str):
    return classify_task.run(text)

agent = Agent(model="openai:gpt-4o-mini", tools=[extract_invoice, classify])
result = agent.run_sync("Extract the data from invoice.pdf and classify the document type.")
print(result.output)

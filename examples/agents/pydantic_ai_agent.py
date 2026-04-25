from pydantic_ai import Agent

from aibackends import configure
from aibackends.tasks import classify, extract_invoice

configure(runtime="llamacpp", model="gemma4-e2b")

agent = Agent(model="openai:gpt-4o-mini", tools=[extract_invoice, classify])
result = agent.run_sync("Extract the data from invoice.pdf and classify the document type.")
print(result.output)

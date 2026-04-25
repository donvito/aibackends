from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from aibackends import configure, register_runtime, reset_config
from aibackends.core.prompting import render_messages_as_text
from aibackends.core.registry import ModelRef
from aibackends.core.runtimes.base import BaseRuntime
from aibackends.core.types import Message, RuntimeConfig, RuntimeResponse


class StubRuntime(BaseRuntime):
    def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        del kwargs
        prompt = render_messages_as_text(messages).lower()
        if schema is None:
            return RuntimeResponse(content="Stub summary", model=self.model_name)

        if schema.__name__ == "Classification":
            labels = {
                "invoice": 0.9 if "invoice" in prompt else 0.1,
                "contract": 0.05,
                "receipt": 0.05,
            }
            payload = {
                "label": max(labels, key=lambda label: labels[label]),
                "confidence": max(labels.values()),
                "all_scores": labels,
            }
        elif schema.__name__ == "InvoiceOutput":
            payload = {
                "vendor": "Acme Corp",
                "line_items": [
                    {
                        "description": "Consulting",
                        "quantity": 1,
                        "unit_price": 1250.0,
                        "amount": 1250.0,
                    }
                ],
                "subtotal": 1250.0,
                "tax": 0.0,
                "total": 1250.0,
                "due_date": None,
                "payment_terms": "Net 30",
            }
        elif schema.__name__ == "SalesCallReport":
            payload = {
                "talk_ratio": {"agent": 0.62, "customer": 0.38},
                "objections": ["price"],
                "buying_signals": ["asked about rollout"],
                "action_items": ["send proposal"],
                "score": 7.4,
                "sentiment": "positive",
            }
        elif schema.__name__ == "VideoAdReport":
            payload = {
                "hook_strength": 8.1,
                "key_messages": ["Fast shipping", "Discount"],
                "cta_clarity": 7.2,
                "emotion_arc": ["curiosity", "confidence", "urgency"],
            }
        else:
            payload = {"name": "Alice", "email": "alice@example.com"}

        return RuntimeResponse(content=json.dumps(payload), model=self.model_name)

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        del kwargs
        return [float(len(text)), 1.0, 0.5]


@pytest.fixture(autouse=True)
def configure_stub_runtime():
    stub_runtime = register_runtime("stub", StubRuntime)
    stub_model = ModelRef(name="stub-model")
    reset_config()
    configure(runtime=stub_runtime, model=stub_model)
    yield RuntimeConfig.model_validate({"runtime": stub_runtime, "model": stub_model})
    reset_config()

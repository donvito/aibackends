from __future__ import annotations

from aibackends.core.types import RuntimeConfig
from aibackends.schemas.invoice import InvoiceOutput
from aibackends.steps._base import StepContext
from aibackends.steps.validate import PydanticValidator


def test_pydantic_validator_accepts_dict_payload():
    validator = PydanticValidator(schema=InvoiceOutput)
    result = validator.run(
        {
            "vendor": "Acme Corp",
            "line_items": [
                {"description": "Consulting", "quantity": 1, "unit_price": 1250.0, "amount": 1250.0}
            ],
            "subtotal": 1250.0,
            "tax": 0.0,
            "total": 1250.0,
            "due_date": None,
            "payment_terms": "Net 30",
        },
        StepContext(task_name="test", runtime_config=RuntimeConfig()),
    )
    assert isinstance(result, InvoiceOutput)
    assert result.vendor == "Acme Corp"

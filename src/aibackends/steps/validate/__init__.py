from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aibackends.steps._base import BaseStep, StepContext
from aibackends.tasks._utils import parse_json_content


class PydanticValidator(BaseStep):
    name = "pydantic_validate"

    def __init__(self, schema: type[BaseModel]) -> None:
        self.schema = schema

    def run(self, payload: Any, context: StepContext) -> BaseModel:
        del context
        if isinstance(payload, self.schema):
            return payload
        if isinstance(payload, BaseModel):
            return self.schema.model_validate(payload.model_dump())
        if isinstance(payload, dict):
            return self.schema.model_validate(payload)
        if isinstance(payload, str):
            return self.schema.model_validate(parse_json_content(payload))
        raise TypeError(f"Unsupported payload type for validation: {type(payload)!r}")

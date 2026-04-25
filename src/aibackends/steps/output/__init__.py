from __future__ import annotations

from typing import Any

from aibackends.steps._base import BaseStep, StepContext


class OutputPassthrough(BaseStep):
    name = "output"

    def run(self, payload: Any, context: StepContext) -> Any:
        del context
        return payload

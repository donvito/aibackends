from __future__ import annotations

from typing import Any

from aibackends.steps._base import BaseStep


class OutputPassthrough(BaseStep):
    name = "output"

    def run(self, payload: Any, context: dict[str, Any]) -> Any:
        del context
        return payload

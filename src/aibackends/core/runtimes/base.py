from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from aibackends.core.types import Message, RuntimeConfig, RuntimeResponse


class BaseRuntime(ABC):
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    @property
    def model_name(self) -> str:
        return self.config.model or self.config.model_path or "unknown"

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        raise NotImplementedError

    async def complete_async(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        return await asyncio.to_thread(self.complete, messages, schema=schema, **kwargs)

    @abstractmethod
    def embed(self, text: str, **kwargs: Any) -> list[float]:
        raise NotImplementedError

    async def embed_async(self, text: str, **kwargs: Any) -> list[float]:
        return await asyncio.to_thread(self.embed, text, **kwargs)

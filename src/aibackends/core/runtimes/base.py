from __future__ import annotations

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from aibackends.core.types import Message, RuntimeConfig, RuntimeResponse


class BaseRuntime(ABC):
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._inference_lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.config.model or self.config.model_path or "unknown"

    def preload(self) -> None:  # noqa: B027 -- intentional no-op default, not abstract
        """Load the model into memory ahead of the first request. No-op by default."""

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

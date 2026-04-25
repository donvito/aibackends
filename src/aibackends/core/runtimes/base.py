from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel

from aibackends.core.exceptions import RuntimeRequestError
from aibackends.core.prompting import normalise_message_content, schema_prompt
from aibackends.core.types import Message, RuntimeConfig, RuntimeResponse, TokenUsage


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    return str(content)


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


class OpenAICompatibleRuntime(BaseRuntime):
    provider_name = "openai-compatible"
    default_base_url = ""
    api_env_var: str | None = None

    def _base_url(self) -> str:
        return (self.config.base_url or self.default_base_url).rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.config.api_key or (os.getenv(self.api_env_var) if self.api_env_var else None)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        data = self._post("/chat/completions", payload)
        content = extract_text_from_content(data["choices"][0]["message"]["content"])
        usage = data.get("usage", {})
        return RuntimeResponse(
            content=content,
            model=data.get("model", self.model_name),
            raw=data,
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            ),
        )

    async def complete_async(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_object"}
        data = await self._post_async("/chat/completions", payload)
        content = extract_text_from_content(data["choices"][0]["message"]["content"])
        usage = data.get("usage", {})
        return RuntimeResponse(
            content=content,
            model=data.get("model", self.model_name),
            raw=data,
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            ),
        )

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        payload = {"model": self.model_name, "input": text}
        data = self._post("/embeddings", payload)
        return [float(value) for value in data["data"][0]["embedding"]]

    async def embed_async(self, text: str, **kwargs: Any) -> list[float]:
        payload = {"model": self.model_name, "input": text}
        data = await self._post_async("/embeddings", payload)
        return [float(value) for value in data["data"][0]["embedding"]]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(
                    f"{self._base_url()}{path}", headers=self._headers(), json=payload
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise RuntimeRequestError(f"{self.provider_name} request failed: {exc}") from exc

    async def _post_async(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                response = await client.post(
                    f"{self._base_url()}{path}",
                    headers=self._headers(),
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise RuntimeRequestError(f"{self.provider_name} request failed: {exc}") from exc


class AnthropicMessagesRuntime(BaseRuntime):
    default_base_url = "https://api.anthropic.com"
    api_env_var = "ANTHROPIC_API_KEY"

    def _headers(self) -> dict[str, str]:
        api_key = self.config.api_key or os.getenv(self.api_env_var)
        return {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": api_key or "",
        }

    def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        system_prompt, clean_messages = self._split_system(messages, schema)
        payload = {
            "model": self.model_name,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "messages": clean_messages,
        }
        if system_prompt:
            payload["system"] = system_prompt
        data = self._post("/v1/messages", payload)
        content = extract_text_from_content(data.get("content", []))
        usage = data.get("usage", {})
        return RuntimeResponse(
            content=content,
            model=data.get("model", self.model_name),
            raw=data,
            usage=TokenUsage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            ),
        )

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        raise RuntimeRequestError(
            "Anthropic does not expose embeddings through this runtime wrapper."
        )

    def _split_system(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None,
    ) -> tuple[str | None, list[Message]]:
        system_parts: list[str] = []
        clean_messages: list[Message] = []
        for message in messages:
            role = message.get("role")
            content = normalise_message_content(message.get("content", ""))
            if role == "system":
                system_parts.append(content)
            else:
                clean_messages.append({"role": role, "content": content})
        if schema is not None:
            system_parts.append(schema_prompt(schema))
        system_prompt = "\n\n".join(part for part in system_parts if part) or None
        return system_prompt, clean_messages

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        base_url = (self.config.base_url or self.default_base_url).rstrip("/")
        try:
            with httpx.Client(timeout=self.config.timeout) as client:
                response = client.post(f"{base_url}{path}", headers=self._headers(), json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as exc:
            raise RuntimeRequestError(f"anthropic request failed: {exc}") from exc

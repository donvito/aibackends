from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aibackends.core.exceptions import RuntimeImportError, RuntimeRequestError
from aibackends.core.model_manager import ModelManager
from aibackends.core.prompting import schema_prompt
from aibackends.core.registry import RuntimeSpec
from aibackends.core.runtimes.base import BaseRuntime
from aibackends.core.types import Message, RuntimeConfig, RuntimeResponse, TokenUsage


class LlamaCppRuntime(BaseRuntime):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        self.model_manager = ModelManager(cache_dir=config.cache_dir)
        self._client: Any | None = None

    def _load_client(self):
        if self._client is not None:
            return self._client
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[llamacpp]' to use the llama.cpp runtime."
            ) from exc

        location = self.model_manager.ensure_model(self.config)
        if not location.local_path:
            raise RuntimeRequestError("llama.cpp requires a local GGUF file.")
        accelerator = self.model_manager.detect_hardware().accelerator
        n_gpu_layers = -1 if accelerator in {"cuda", "metal"} else 0
        self._client = Llama(
            model_path=location.local_path,
            n_ctx=self.config.extra_options.get("n_ctx", 8192),
            n_gpu_layers=self.config.extra_options.get("n_gpu_layers", n_gpu_layers),
            verbose=False,
        )
        return self._client

    def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        client = self._load_client()
        payload_messages = list(messages)
        if schema is not None:
            payload_messages = [
                {"role": "system", "content": schema_prompt(schema)},
                *payload_messages,
            ]
        response = client.create_chat_completion(
            messages=payload_messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            response_format={"type": "json_object"} if schema is not None else None,
        )
        choice = response["choices"][0]["message"]
        usage = response.get("usage", {})
        return RuntimeResponse(
            content=str(choice.get("content", "")),
            model=response.get("model", self.model_name),
            raw=response,
            usage=TokenUsage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            ),
        )

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        client = self._load_client()
        if not hasattr(client, "create_embedding"):
            raise RuntimeRequestError("This llama.cpp build does not support embeddings.")
        data = client.create_embedding(text)
        embedding = data["data"][0]["embedding"]
        return [float(value) for value in embedding]


RUNTIME_SPEC = RuntimeSpec(name="llamacpp", factory=LlamaCppRuntime)

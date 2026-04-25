from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from aibackends.core.exceptions import RuntimeImportError
from aibackends.core.model_manager import ModelManager
from aibackends.core.model_registry import apply_transformer_model_profile
from aibackends.core.prompting import PromptRenderer
from aibackends.core.registry import RuntimeSpec
from aibackends.core.runtimes.base import BaseRuntime
from aibackends.core.types import Message, RuntimeConfig, RuntimeResponse


class TransformersRuntime(BaseRuntime):
    def __init__(self, config: RuntimeConfig) -> None:
        effective_config = apply_transformer_model_profile(config)
        super().__init__(effective_config)
        self.model_manager = ModelManager(cache_dir=effective_config.cache_dir)
        self.prompt_renderer = PromptRenderer(effective_config)
        self._tokenizer: Any | None = None
        self._generator: Any | None = None
        self._embed_model: Any | None = None
        self._embed_tokenizer: Any | None = None
        self._torch: Any | None = None

    def _model_id(self) -> str:
        location = self.model_manager.ensure_model(self.config)
        return location.local_path or location.source

    def _load_generator(self):
        if self._generator is not None and self._tokenizer is not None:
            return self._tokenizer, self._generator
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[transformers]' to use the Transformers runtime."
            ) from exc

        model_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if self.config.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        model: Any = AutoModelForCausalLM.from_pretrained(
            self._model_id(),
            device_map=self.config.device or "auto",
            **model_kwargs,
        )
        if self.config.adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeImportError(
                    "Install 'peft' to load adapters in the Transformers runtime."
                ) from exc
            model = PeftModel.from_pretrained(model, self.config.adapter)

        tokenizer = AutoTokenizer.from_pretrained(self._model_id(), trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(model, "eval"):
            model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._generator = model
        return tokenizer, model

    def _load_embedder(self):
        if self._embed_model is not None and self._embed_tokenizer is not None:
            return self._embed_tokenizer, self._embed_model
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[transformers]' to enable embeddings."
            ) from exc

        self._embed_tokenizer = AutoTokenizer.from_pretrained(
            self._model_id(), trust_remote_code=True
        )
        self._embed_model = AutoModel.from_pretrained(self._model_id(), trust_remote_code=True)
        return self._embed_tokenizer, self._embed_model

    def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        tokenizer, model = self._load_generator()
        prompt_result = self.prompt_renderer.render(messages, tokenizer=tokenizer, schema=schema)
        prompt = prompt_result.prompt
        inputs = tokenizer(prompt, return_tensors="pt")
        temperature = kwargs.get(
            "temperature",
            self.config.extra_options.get("temperature", self.config.temperature),
        )
        if temperature is None:
            temperature = self.config.temperature
        device = getattr(model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}

        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": kwargs.get(
                "max_tokens", self.config.extra_options.get("max_tokens", self.config.max_tokens)
            ),
            "temperature": temperature,
            "do_sample": float(temperature) > 0,
        }
        if tokenizer.pad_token_id is not None:
            generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token_id is not None:
            generate_kwargs["eos_token_id"] = eos_token_id
        output = model.generate(**generate_kwargs)
        input_length = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][input_length:]
        content = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return RuntimeResponse(
            content=content,
            model=self.model_name,
            raw={
                "prompt": prompt,
                "prompt_format": prompt_result.format_used,
                "template_source": prompt_result.template_source,
            },
        )

    def embed(self, text: str, **kwargs: Any) -> list[float]:
        tokenizer, model = self._load_embedder()
        if self._torch is None:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeImportError("Install 'torch' to compute local embeddings.") from exc
            self._torch = torch

        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        with self._torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return [float(value) for value in pooled[0].cpu().tolist()]


RUNTIME_SPEC = RuntimeSpec(name="transformers", factory=TransformersRuntime)

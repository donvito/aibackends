from __future__ import annotations

import base64
import mimetypes
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel

from aibackends.core.exceptions import RuntimeImportError, RuntimeRequestError
from aibackends.core.model_manager import ModelLocation, ModelManager
from aibackends.core.prompting import (
    build_prompt_messages,
    normalise_message_content,
    schema_prompt,
)
from aibackends.core.registry import RuntimeSpec
from aibackends.core.runtimes.base import BaseRuntime
from aibackends.core.types import Message, RuntimeConfig, RuntimeResponse, TokenUsage

IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def has_image_inputs(messages: list[Message]) -> bool:
    for message in messages:
        if _content_has_images(message.get("content")):
            return True
    return False


def build_llamacpp_multimodal_messages(
    messages: list[Message],
    *,
    schema: type[BaseModel] | None = None,
    merge_system_into_first_user: bool = False,
) -> list[Message]:
    prompt_messages = build_prompt_messages(messages, schema=schema)
    merged_messages = (
        _merge_system_into_first_user(prompt_messages)
        if merge_system_into_first_user
        else prompt_messages
    )
    normalized: list[Message] = []
    for message in merged_messages:
        role = str(message.get("role", "user"))
        content: Any
        if role == "user":
            content = _normalise_multimodal_content(message.get("content"))
        else:
            content = normalise_message_content(message.get("content", ""))
        normalized.append(
            {
                "role": role,
                "content": content,
            }
        )
    return normalized


def _merge_system_into_first_user(messages: list[Message]) -> list[Message]:
    merged_messages = [dict(message) for message in messages]
    if not merged_messages or merged_messages[0].get("role") != "system":
        return merged_messages

    system_text = normalise_message_content(merged_messages[0].get("content", "")).strip()
    remaining = merged_messages[1:]
    if not system_text:
        return remaining

    for message in remaining:
        if message.get("role") != "user":
            continue
        parts = _normalise_multimodal_content(message.get("content"))
        if parts and parts[0].get("type") == "text":
            first_text = str(parts[0].get("text", "")).strip()
            parts[0]["text"] = f"{system_text}\n\n{first_text}" if first_text else system_text
        else:
            parts.insert(0, {"type": "text", "text": system_text})
        message["content"] = parts
        return remaining

    return [{"role": "user", "content": [{"type": "text", "text": system_text}]}, *remaining]


def _content_has_images(content: Any) -> bool:
    items = content if isinstance(content, list) else [content]
    for item in items:
        if isinstance(item, Path):
            return True
        if not isinstance(item, dict):
            continue
        if item.get("type") in {"image", "image_url"}:
            return True
        if "image" in item or "image_url" in item:
            return True
    return False


def _normalise_multimodal_content(content: Any) -> list[dict[str, Any]]:
    items = content if isinstance(content, list) else [content]
    parts: list[dict[str, Any]] = []
    for item in items:
        part = _normalise_multimodal_part(item)
        if part is not None:
            parts.append(part)
    return parts


def _normalise_multimodal_part(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, Path):
        return _build_image_part(item)
    if isinstance(item, str):
        return {"type": "text", "text": item}
    if not isinstance(item, dict):
        return {"type": "text", "text": str(item)}

    if item.get("type") in {"image", "image_url"} or "image" in item or "image_url" in item:
        source = _extract_image_source(item)
        if source is None:
            return None
        return _build_image_part(source)

    if item.get("type") == "text" or "text" in item:
        return {"type": "text", "text": str(item.get("text", ""))}
    return {"type": "text", "text": str(item)}


def _extract_image_source(item: dict[str, Any]) -> Any:
    source = item.get("image", item.get("image_url"))
    if isinstance(source, dict):
        return source.get("url")
    return source


def _build_image_part(source: Any) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": _coerce_image_url(source)},
    }


def _coerce_image_url(source: Any) -> str:
    if isinstance(source, Path):
        return image_path_to_data_uri(source)

    source_text = str(source)
    if source_text.startswith("data:"):
        return source_text

    parsed = urlparse(source_text)
    if parsed.scheme in {"http", "https"}:
        return source_text
    if parsed.scheme == "file":
        return image_path_to_data_uri(Path(parsed.path))

    candidate = Path(source_text).expanduser()
    if candidate.exists():
        return image_path_to_data_uri(candidate)
    if candidate.suffix.lower() in IMAGE_SUFFIXES:
        raise RuntimeRequestError(f"Image path does not exist: {candidate}")
    return source_text


def image_path_to_data_uri(path: str | Path) -> str:
    image_path = Path(path).expanduser()
    if not image_path.exists():
        raise RuntimeRequestError(f"Image path does not exist: {image_path}")
    mime_type, _ = mimetypes.guess_type(str(image_path))
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type or 'image/png'};base64,{encoded}"


class LlamaCppRuntime(BaseRuntime):
    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__(config)
        self.model_manager = ModelManager(cache_dir=config.cache_dir)
        self._client: Any | None = None
        self._multimodal_client: Any | None = None

    def _load_client(self) -> Any:
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
        self._client = Llama(**self._build_client_kwargs(location))
        return self._client

    def _load_multimodal_client(self) -> Any:
        if self._multimodal_client is not None:
            return self._multimodal_client
        location = self.model_manager.ensure_model(self.config)
        family = self._multimodal_family(location)
        if family is None:
            raise RuntimeRequestError(
                "Image inputs in the llama.cpp runtime are currently supported for "
                "Gemma and Qwen VL GGUF models only."
            )

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[llamacpp]' to use the llama.cpp runtime."
            ) from exc

        if not location.local_path:
            raise RuntimeRequestError("llama.cpp requires a local GGUF file.")
        mmproj_path = self._resolve_mmproj_path(location)
        chat_handler = self._build_multimodal_chat_handler(family, mmproj_path)
        self._multimodal_client = Llama(
            **self._build_client_kwargs(location),
            chat_handler=chat_handler,
        )
        return self._multimodal_client

    def _build_client_kwargs(self, location: ModelLocation) -> dict[str, Any]:
        if not location.local_path:
            raise RuntimeRequestError("llama.cpp requires a local GGUF file.")

        accelerator = self.model_manager.detect_hardware().accelerator
        n_gpu_layers = -1 if accelerator in {"cuda", "metal"} else 0
        options: dict[str, Any] = {
            "model_path": location.local_path,
            "n_ctx": self.config.extra_options.get("n_ctx", 8192),
            "n_gpu_layers": self.config.extra_options.get("n_gpu_layers", n_gpu_layers),
            "verbose": False,
        }
        if "gemma" in self.model_name.lower():
            options["chat_format"] = self.config.extra_options.get("chat_format", "gemma")
        for option in ("flash_attn", "n_batch", "n_ubatch"):
            if option in self.config.extra_options:
                options[option] = self.config.extra_options[option]
        return options

    def _multimodal_family(self, location: ModelLocation | None = None) -> str | None:
        model_reference = f"{self.model_name} {self.config.model_path or ''}".lower()
        if location is not None:
            model_reference = f"{model_reference} {location.source}".lower()
        if "gemma" in model_reference:
            return "gemma"
        if "qwen3-vl" in model_reference or "qwen2.5-vl" in model_reference:
            return "qwen-vl"
        return None

    def _resolve_mmproj_path(self, location: ModelLocation) -> str:
        mmproj_override = self.config.extra_options.get("mmproj_path")
        if mmproj_override is not None:
            mmproj_path = Path(str(mmproj_override)).expanduser()
            if not mmproj_path.exists():
                raise RuntimeRequestError(f"mmproj path does not exist: {mmproj_path}")
            return str(mmproj_path)

        if location.local_path:
            model_path = Path(location.local_path)
            local_mmproj = self._find_local_mmproj(model_path)
            if local_mmproj is not None:
                return str(local_mmproj)

        if self._looks_like_repo_id(location.source):
            return str(self._download_mmproj(location.source))

        raise RuntimeRequestError(
            "Multimodal llama.cpp inference requires a matching mmproj GGUF. "
            "Set `extra_options={'mmproj_path': '/path/to/mmproj.gguf'}` or place a "
            "`mmproj*.gguf` file beside the model."
        )

    def _find_local_mmproj(self, model_path: Path) -> Path | None:
        candidates = sorted(model_path.parent.glob("mmproj*.gguf"))
        if not candidates:
            return None
        return self._select_local_mmproj(candidates)

    def _select_local_mmproj(self, candidates: list[Path]) -> Path:
        preferred_order = ("bf16", "f16")
        for quant in preferred_order:
            for candidate in candidates:
                if quant in candidate.name.lower():
                    return candidate
        return candidates[0]

    def _download_mmproj(self, repo_id: str) -> Path:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[llamacpp]' to download the matching mmproj file."
            ) from exc

        candidates = [
            PurePosixPath(repo_file)
            for repo_file in list_repo_files(repo_id)
            if repo_file.lower().endswith(".gguf")
            and PurePosixPath(repo_file).name.lower().startswith("mmproj")
        ]
        if not candidates:
            raise RuntimeRequestError(
                f"No mmproj GGUF files were found in repository: {repo_id}. "
                "Provide `extra_options['mmproj_path']` with the matching projector."
            )

        selected = self._select_repo_mmproj(candidates)
        subfolder = None if selected.parent == PurePosixPath(".") else selected.parent.as_posix()
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=selected.name,
            subfolder=subfolder,
            cache_dir=self.model_manager._hf_cache_dir(),
        )
        return Path(local_path)

    def _select_repo_mmproj(self, candidates: list[PurePosixPath]) -> PurePosixPath:
        preferred_order = ("bf16", "f16")
        for quant in preferred_order:
            for candidate in candidates:
                if quant in candidate.name.lower():
                    return candidate
        return sorted(candidates, key=lambda item: item.as_posix().lower())[0]

    def _looks_like_repo_id(self, value: str) -> bool:
        return (
            "/" in value
            and not value.startswith(("/", ".", "~"))
            and not value.lower().endswith(".gguf")
        )

    def _build_multimodal_chat_handler(self, family: str, mmproj_path: str) -> Any:
        if family == "gemma":
            return self._build_gemma_vision_chat_handler(mmproj_path)
        if family == "qwen-vl":
            return self._build_qwen_vl_chat_handler(mmproj_path)
        raise RuntimeRequestError(f"Unsupported multimodal llama.cpp family: {family}")

    def _build_gemma_vision_chat_handler(self, mmproj_path: str) -> Any:
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError as exc:
            raise RuntimeImportError(
                "Install a recent 'llama-cpp-python' build with multimodal chat handlers."
            ) from exc

        class GemmaVisionChatHandler(Llava15ChatHandler):
            DEFAULT_SYSTEM_MESSAGE = None
            CHAT_FORMAT = (
                "{% for message in messages %}"
                "{% if message.role == 'user' %}"
                "<start_of_turn>user\n"
                "{% if message.content is string %}"
                "{{ message.content }}"
                "{% endif %}"
                "{% if message.content is iterable and message.content is not string %}"
                "{% for content in message.content %}"
                "{% if content.type == 'image_url' and content.image_url is string %}"
                "{{ content.image_url }}"
                "{% endif %}"
                "{% if content.type == 'image_url' and content.image_url is mapping %}"
                "{{ content.image_url.url }}"
                "{% endif %}"
                "{% if content.type == 'text' %}"
                "{{ content.text }}"
                "{% endif %}"
                "{% endfor %}"
                "{% endif %}"
                "<end_of_turn>\n"
                "{% endif %}"
                "{% if message.role == 'assistant' and message.content is not none %}"
                "<start_of_turn>model\n{{ message.content }}<end_of_turn>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
            )

        return GemmaVisionChatHandler(clip_model_path=mmproj_path, verbose=False)

    def _build_qwen_vl_chat_handler(self, mmproj_path: str) -> Any:
        try:
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
        except ImportError as exc:
            raise RuntimeImportError(
                "Install a recent 'llama-cpp-python' build with Qwen VL chat handler support."
            ) from exc

        return Qwen25VLChatHandler(clip_model_path=mmproj_path, verbose=False)

    def complete(
        self,
        messages: list[Message],
        schema: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> RuntimeResponse:
        if has_image_inputs(messages):
            family = self._multimodal_family()
            client = self._load_multimodal_client()
            payload_messages = build_llamacpp_multimodal_messages(
                messages,
                schema=schema,
                merge_system_into_first_user=family == "gemma",
            )
        else:
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

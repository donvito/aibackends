from __future__ import annotations

import base64
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

from aibackends.core.model_manager import ModelLocation
from aibackends.core.runtimes.llamacpp import (
    LlamaCppRuntime,
    build_llamacpp_multimodal_messages,
)
from aibackends.core.types import RuntimeConfig


class VisionResult(BaseModel):
    description: str


class FakeMultimodalClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.last_kwargs = kwargs
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '{"description": "receipt"}',
                    }
                }
            ],
            "model": "gemma-4-e2b",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }


def test_build_llamacpp_multimodal_messages_inlines_system_prompt_and_image(tmp_path):
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nreceipt")

    messages = build_llamacpp_multimodal_messages(
        [
            {"role": "system", "content": "Return OCR-ready JSON."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": str(image_path)}},
                ],
            },
        ],
        schema=VisionResult,
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"

    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert "Return exactly one JSON object" in content[0]["text"]
    assert "Return OCR-ready JSON." in content[0]["text"]
    assert "Describe this image." in content[0]["text"]
    assert content[1]["type"] == "image_url"

    image_url = content[1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    encoded = image_url.split(",", 1)[1]
    assert base64.b64decode(encoded).startswith(b"\x89PNG\r\n\x1a\n")


def test_llamacpp_runtime_uses_multimodal_client_for_image_messages(monkeypatch, tmp_path):
    image_path = tmp_path / "receipt.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nreceipt")

    runtime = LlamaCppRuntime(RuntimeConfig(runtime="llamacpp", model="gemma4-e2b"))
    client = FakeMultimodalClient()
    monkeypatch.setattr(runtime, "_load_multimodal_client", lambda: client)

    response = runtime.complete(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": str(image_path)}},
                ],
            }
        ],
        schema=VisionResult,
    )

    assert client.last_kwargs is not None
    assert client.last_kwargs["response_format"] == {"type": "json_object"}
    payload_messages = client.last_kwargs["messages"]
    assert len(payload_messages) == 1
    assert payload_messages[0]["role"] == "user"
    assert response.content == '{"description": "receipt"}'
    assert response.usage.input_tokens == 11
    assert response.usage.output_tokens == 7


def test_resolve_mmproj_path_uses_explicit_override(tmp_path):
    model_path = tmp_path / "gemma-4.gguf"
    model_path.write_text("weights")
    mmproj_path = tmp_path / "mmproj-BF16.gguf"
    mmproj_path.write_text("projector")

    runtime = LlamaCppRuntime(
        RuntimeConfig(
            runtime="llamacpp",
            model="gemma4-e2b",
            extra_options={"mmproj_path": str(mmproj_path)},
        )
    )

    resolved = runtime._resolve_mmproj_path(
        ModelLocation(
            source="unsloth/gemma-4-E2B-it-GGUF",
            local_path=str(model_path),
        )
    )

    assert resolved == str(mmproj_path)


def test_resolve_mmproj_path_downloads_matching_projector(monkeypatch, tmp_path):
    model_path = tmp_path / "gemma-4-E2B-it-Q4_K_M.gguf"
    model_path.write_text("weights")
    cache_dir = tmp_path / "models"
    downloads: list[dict[str, str | None]] = []

    def fake_list_repo_files(repo_id: str) -> list[str]:
        assert repo_id == "unsloth/gemma-4-E2B-it-GGUF"
        return ["README.md", "mmproj-F16.gguf", "mmproj-BF16.gguf"]

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        subfolder: str | None = None,
        cache_dir: str | None = None,
    ) -> str:
        downloads.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "subfolder": subfolder,
                "cache_dir": cache_dir,
            }
        )
        target = Path(cache_dir or tmp_path) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("projector")
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            list_repo_files=fake_list_repo_files,
            hf_hub_download=fake_hf_hub_download,
        ),
    )

    runtime = LlamaCppRuntime(
        RuntimeConfig(runtime="llamacpp", model="gemma4-e2b", cache_dir=str(cache_dir))
    )

    resolved = runtime._resolve_mmproj_path(
        ModelLocation(
            source="unsloth/gemma-4-E2B-it-GGUF",
            local_path=str(model_path),
        )
    )

    assert resolved == str(cache_dir / "mmproj-BF16.gguf")
    assert downloads == [
        {
            "repo_id": "unsloth/gemma-4-E2B-it-GGUF",
            "filename": "mmproj-BF16.gguf",
            "subfolder": None,
            "cache_dir": str(cache_dir),
        }
    ]

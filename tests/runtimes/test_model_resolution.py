from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

import aibackends.core.model_manager as model_manager_module
from aibackends.core.model_manager import ModelManager
from aibackends.core.model_registry import resolve_model_alias
from aibackends.core.types import RuntimeConfig


def test_resolve_model_alias_uses_llamacpp_gemma_overrides():
    assert resolve_model_alias("gemma4-e4b", runtime="llamacpp") == (
        "unsloth/gemma-4-E4B-it-GGUF"
    )
    assert resolve_model_alias("gemma4-e2b", runtime="llamacpp") == (
        "unsloth/gemma-4-E2B-it-GGUF"
    )


def test_resolve_model_alias_falls_back_to_shared_registry():
    assert resolve_model_alias("gemma4-e4b", runtime="transformers") == "google/gemma-4-E4B-it"
    assert resolve_model_alias("bge-small", runtime="llamacpp") == "BAAI/bge-small-en-v1.5"
    assert resolve_model_alias("custom/model", runtime="llamacpp") == "custom/model"


def test_model_manager_resolve_model_name_uses_runtime_aliases(tmp_path):
    manager = ModelManager(cache_dir=str(tmp_path / "models"))

    llama_config = RuntimeConfig(runtime="llamacpp", model="gemma4-e4b")
    transformers_config = RuntimeConfig(runtime="transformers", model="gemma4-e4b")

    assert manager.resolve_model_name(llama_config) == "unsloth/gemma-4-E4B-it-GGUF"
    assert manager.resolve_model_name(transformers_config) == "google/gemma-4-E4B-it"


def test_detect_hardware_avoids_torch_import_on_apple_silicon(monkeypatch):
    attempts: list[str] = []
    real_import = builtins.__import__

    def tracking_import(name: str, *args, **kwargs):
        if name == "torch":
            attempts.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(model_manager_module.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(model_manager_module.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(builtins, "__import__", tracking_import)

    manager = ModelManager()
    hardware = manager.detect_hardware()

    assert hardware.accelerator == "metal"
    assert attempts == []


def test_download_gguf_repo_downloads_only_selected_quant(monkeypatch, tmp_path):
    calls: list[dict[str, str | None]] = []

    def fake_list_repo_files(repo_id: str) -> list[str]:
        assert repo_id == "repo/model"
        return [
            "README.md",
            "mmproj-model.gguf",
            "model-imatrix.gguf",
            "model-Q4_K_M.gguf",
            "model-Q5_K_M.gguf",
        ]

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        subfolder: str | None = None,
        cache_dir: str | None = None,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "subfolder": subfolder,
                "cache_dir": cache_dir,
            }
        )
        base_dir = Path(cache_dir or tmp_path / "hf-cache")
        target = base_dir / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("weights")
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            list_repo_files=fake_list_repo_files,
            hf_hub_download=fake_hf_hub_download,
        ),
    )

    manager = ModelManager(cache_dir=str(tmp_path / "models"))
    monkeypatch.setattr(manager, "default_quantization", lambda: "Q4_K_M")

    local_path = manager._download_gguf_repo("repo/model")

    assert local_path.name == "model-Q4_K_M.gguf"
    assert calls == [
        {
            "repo_id": "repo/model",
            "filename": "model-Q4_K_M.gguf",
            "subfolder": None,
            "cache_dir": str(tmp_path / "models"),
        }
    ]


def test_download_gguf_repo_passes_subfolder_for_nested_files(monkeypatch, tmp_path):
    calls: list[dict[str, str | None]] = []

    def fake_list_repo_files(repo_id: str) -> list[str]:
        assert repo_id == "repo/model"
        return ["quantized/model-Q4_K_M.gguf"]

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        subfolder: str | None = None,
        cache_dir: str | None = None,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "subfolder": subfolder,
                "cache_dir": cache_dir,
            }
        )
        base_dir = Path(cache_dir or tmp_path / "hf-cache")
        target = base_dir / (subfolder or "") / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("weights")
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            list_repo_files=fake_list_repo_files,
            hf_hub_download=fake_hf_hub_download,
        ),
    )

    manager = ModelManager(cache_dir=str(tmp_path / "models"))
    monkeypatch.setattr(manager, "default_quantization", lambda: "Q4_K_M")

    local_path = manager._download_gguf_repo("repo/model")

    assert local_path == tmp_path / "models" / "quantized" / "model-Q4_K_M.gguf"
    assert calls == [
        {
            "repo_id": "repo/model",
            "filename": "model-Q4_K_M.gguf",
            "subfolder": "quantized",
            "cache_dir": str(tmp_path / "models"),
        }
    ]


def test_download_gguf_repo_uses_hf_default_cache_when_unset(monkeypatch, tmp_path):
    calls: list[dict[str, str | None]] = []

    def fake_list_repo_files(repo_id: str) -> list[str]:
        assert repo_id == "repo/model"
        return ["model-Q4_K_M.gguf"]

    def fake_hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        subfolder: str | None = None,
        cache_dir: str | None = None,
    ) -> str:
        calls.append(
            {
                "repo_id": repo_id,
                "filename": filename,
                "subfolder": subfolder,
                "cache_dir": cache_dir,
            }
        )
        target = tmp_path / "hf-default" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("weights")
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            list_repo_files=fake_list_repo_files,
            hf_hub_download=fake_hf_hub_download,
        ),
    )

    manager = ModelManager()
    monkeypatch.setattr(manager, "default_quantization", lambda: "Q4_K_M")

    local_path = manager._download_gguf_repo("repo/model")

    assert local_path == tmp_path / "hf-default" / "model-Q4_K_M.gguf"
    assert calls == [
        {
            "repo_id": "repo/model",
            "filename": "model-Q4_K_M.gguf",
            "subfolder": None,
            "cache_dir": None,
        }
    ]


def test_pull_model_uses_snapshot_download_cache_dir(monkeypatch, tmp_path):
    calls: list[dict[str, str | None]] = []

    def fake_snapshot_download(repo_id: str, *, cache_dir: str | None = None) -> str:
        calls.append({"repo_id": repo_id, "cache_dir": cache_dir})
        target = (
            Path(cache_dir or tmp_path / "hf-default")
            / "snapshots"
            / repo_id.replace("/", "--")
        )
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    manager = ModelManager(cache_dir=str(tmp_path / "models"))
    location = manager.pull_model(
        RuntimeConfig(runtime="transformers", model="google/gemma-4-E2B-it")
    )

    assert location.local_path == str(tmp_path / "models" / "snapshots" / "google--gemma-4-E2B-it")
    assert calls == [
        {
            "repo_id": "google/gemma-4-E2B-it",
            "cache_dir": str(tmp_path / "models"),
        }
    ]

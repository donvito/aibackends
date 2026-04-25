from __future__ import annotations

import platform
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from aibackends.core.exceptions import ModelResolutionError, RuntimeImportError
from aibackends.core.model_registry import resolve_model_alias
from aibackends.core.types import RuntimeConfig
from aibackends.model_support import get_model_support


@dataclass(slots=True)
class HardwareProfile:
    accelerator: str
    machine: str


@dataclass(slots=True)
class ModelLocation:
    source: str
    local_path: str | None = None


class ModelManager:
    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def detect_hardware(self) -> HardwareProfile:
        machine = platform.machine().lower()
        if platform.system() == "Darwin" and machine in {"arm64", "aarch64"}:
            return HardwareProfile(accelerator="metal", machine=machine)

        accelerator = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                accelerator = "cuda"
        except Exception:
            pass
        return HardwareProfile(accelerator=accelerator, machine=machine)

    def default_quantization(self) -> str:
        return "Q5_K_M" if self.detect_hardware().accelerator in {"cuda", "metal"} else "Q4_K_M"

    def resolve_model_name(self, config: RuntimeConfig) -> str:
        if config.model_path:
            path = Path(config.model_path).expanduser()
            if not path.exists():
                raise ModelResolutionError(f"Model path does not exist: {path}")
            return str(path)
        resolved = resolve_model_alias(config.model, runtime=config.runtime)
        if not resolved:
            raise ModelResolutionError("No model or model_path configured.")
        return resolved

    def ensure_model(self, config: RuntimeConfig) -> ModelLocation:
        resolved = self.resolve_model_name(config)
        candidate = Path(resolved).expanduser()
        if candidate.exists():
            return ModelLocation(source=resolved, local_path=str(candidate))

        support = get_model_support(config.runtime)
        if support is not None and support.ensure_model is not None:
            return support.ensure_model(self, config, resolved)

        return ModelLocation(source=resolved, local_path=None)

    def pull_model(self, config: RuntimeConfig) -> ModelLocation:
        resolved = self.resolve_model_name(config)
        support = get_model_support(config.runtime)
        if support is not None and support.pull_model is not None:
            return support.pull_model(self, config, resolved)

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'huggingface_hub' to pre-pull models with `aibackends pull`."
            ) from exc

        local_dir = snapshot_download(repo_id=resolved, **self._hf_cache_kwargs())
        return ModelLocation(source=resolved, local_path=local_dir)

    def _download_gguf_repo(self, repo_id: str) -> Path:
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as exc:
            raise RuntimeImportError(
                "Install 'aibackends[llamacpp]' to enable GGUF model download and caching."
            ) from exc

        candidates = self._list_gguf_files(repo_id, list_repo_files)
        if not candidates:
            raise ModelResolutionError(
                f"No GGUF files found in repository: {repo_id}. "
                "For llama.cpp, use a GGUF repo ID or a local GGUF file."
            )

        selected = self._select_gguf_file(candidates)
        subfolder = None if selected.parent == PurePosixPath(".") else selected.parent.as_posix()
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=selected.name,
            subfolder=subfolder,
            **self._hf_cache_kwargs(),
        )
        return Path(local_path)

    def _hf_cache_kwargs(self) -> dict[str, str]:
        if self.cache_dir is None:
            return {}
        return {"cache_dir": str(self.cache_dir)}

    def _list_gguf_files(
        self,
        repo_id: str,
        list_repo_files: Callable[[str], list[str]],
    ) -> list[PurePosixPath]:
        candidates = [
            PurePosixPath(repo_file)
            for repo_file in list_repo_files(repo_id)
            if repo_file.lower().endswith(".gguf") and not self._is_auxiliary_gguf(repo_file)
        ]
        return sorted(candidates, key=lambda item: item.as_posix().lower())

    def _select_gguf_file(self, candidates: list[PurePosixPath]) -> PurePosixPath:
        preferred_order = [
            self.default_quantization(),
            "Q4_K_M",
            "Q5_K_M",
            "Q4_K_S",
            "Q5_K_S",
            "Q6_K",
            "Q6_K_L",
            "Q5_K_L",
            "Q4_K_L",
            "IQ4_XS",
            "IQ4_NL",
            "Q8_0",
            "BF16",
            "F16",
        ]
        ordered_preferences: list[str] = []
        for quant in preferred_order:
            if quant not in ordered_preferences:
                ordered_preferences.append(quant)

        for quant in ordered_preferences:
            quant_lower = quant.lower()
            for candidate in candidates:
                if quant_lower in candidate.name.lower():
                    return candidate

        quantized = [
            candidate
            for candidate in candidates
            if "-q" in candidate.name.lower() or "-iq" in candidate.name.lower()
        ]
        if quantized:
            return quantized[0]
        return candidates[0]

    def _is_auxiliary_gguf(self, repo_file: str) -> bool:
        filename = PurePosixPath(repo_file).name.lower()
        return filename.startswith("mmproj-") or "imatrix" in filename

from __future__ import annotations

MODEL_REGISTRY: dict[str, str] = {
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
    "qwen3-vl-4b": "bartowski/Qwen3-VL-4B-Instruct-GGUF",
    "qwen3-vl-8b": "bartowski/Qwen3-VL-7B-Instruct-GGUF",
    "gemma3-4b": "bartowski/gemma-3-4b-it-GGUF",
    "gemma3-12b": "bartowski/gemma-3-12b-it-GGUF",
    "llama3-8b": "bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    "mistral-7b": "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "phi4-mini": "bartowski/phi-4-mini-instruct-GGUF",
    "minilm-l6": "sentence-transformers/all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "openai-privacy": "openai/privacy-filter",
}

RUNTIME_MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "llamacpp": {
        "gemma4-e2b": "unsloth/gemma-4-E2B-it-GGUF",
        "gemma4-e4b": "unsloth/gemma-4-E4B-it-GGUF",
    }
}


def resolve_model_alias(name: str | None, *, runtime: str | None = None) -> str | None:
    if not name:
        return None
    if runtime and name in RUNTIME_MODEL_REGISTRY.get(runtime, {}):
        return RUNTIME_MODEL_REGISTRY[runtime][name]
    return MODEL_REGISTRY.get(name, name)

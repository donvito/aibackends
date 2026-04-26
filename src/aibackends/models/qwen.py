from aibackends.core.registry import TransformerModelProfile

MODEL_PROFILES = [
    TransformerModelProfile(
        name="qwen3-vl-4b",
        model_id="Qwen/Qwen3-VL-4B-Instruct-GGUF",
        runtime="llamacpp",
    ),
    TransformerModelProfile(
        name="qwen3-vl-8b",
        model_id="Qwen/Qwen3-VL-8B-Instruct-GGUF",
        runtime="llamacpp",
    ),
]

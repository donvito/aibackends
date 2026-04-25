from aibackends.core.registry import TransformerModelProfile

MODEL_PROFILES = [
    TransformerModelProfile(
        name="qwen3-vl-4b",
        model_id="bartowski/Qwen3-VL-4B-Instruct-GGUF",
        runtime=None,
    ),
    TransformerModelProfile(
        name="qwen3-vl-8b",
        model_id="bartowski/Qwen3-VL-7B-Instruct-GGUF",
        runtime=None,
    ),
]

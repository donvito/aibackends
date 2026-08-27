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
    # Qwen3.8-27B (Apache 2.0, Aug 2026): dense multimodal open model; the
    # community GGUF build runs on llama.cpp (~15.4 GB at Q4_K_M).
    TransformerModelProfile(
        name="qwen3.8-27b",
        aliases=("qwen38-27b",),
        model_id="unsloth/Qwen3.8-27B-GGUF",
        runtime="llamacpp",
        quantization="Q4_K_M",
    ),
]

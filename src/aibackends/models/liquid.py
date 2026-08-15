from aibackends.core.registry import TransformerModelProfile

# LiquidAI recommends temperature=0.1, top_k=50, repetition_penalty=1.1 for
# LFM2.5 models (https://huggingface.co/LiquidAI/LFM2.5-2.6B).
_LFM25_GENERATION_DEFAULTS = {
    "temperature": 0.1,
    "top_k": 50,
    "repetition_penalty": 1.1,
}

MODEL_PROFILES = [
    TransformerModelProfile(
        name="lfm2.5-2.6b",
        aliases=("lfm25-2.6b",),
        model_id="LiquidAI/LFM2.5-2.6B",
        runtime="transformers",
        generation_defaults={
            **_LFM25_GENERATION_DEFAULTS,
            "dtype": "bfloat16",
        },
    ),
    TransformerModelProfile(
        name="lfm2.5-2.6b",
        aliases=("lfm25-2.6b",),
        model_id="LiquidAI/LFM2.5-2.6B-GGUF",
        runtime="llamacpp",
        quantization="Q4_K_M",
        generation_defaults=_LFM25_GENERATION_DEFAULTS,
    ),
    # Vision-language model; image inputs run through the llama.cpp multimodal
    # path (main GGUF + mmproj projector downloaded from the same repo).
    TransformerModelProfile(
        name="lfm2.5-vl-3b",
        aliases=("lfm25-vl-3b",),
        model_id="LiquidAI/LFM2.5-VL-3B-GGUF",
        runtime="llamacpp",
        quantization="Q4_K_M",
        generation_defaults=_LFM25_GENERATION_DEFAULTS,
    ),
]

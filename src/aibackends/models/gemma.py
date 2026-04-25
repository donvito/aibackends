from aibackends.core.registry import TransformerModelProfile

MODEL_PROFILES = [
    TransformerModelProfile(
        name="gemma4-e2b",
        model_id="google/gemma-4-E2B-it",
        runtime="transformers",
    ),
    TransformerModelProfile(
        name="gemma4-e4b",
        model_id="google/gemma-4-E4B-it",
        runtime="transformers",
    ),
    TransformerModelProfile(
        name="gemma4-e2b",
        model_id="unsloth/gemma-4-E2B-it-GGUF",
        runtime="llamacpp",
    ),
    TransformerModelProfile(
        name="gemma4-e4b",
        model_id="unsloth/gemma-4-E4B-it-GGUF",
        runtime="llamacpp",
    ),
    TransformerModelProfile(
        name="gemma3-4b",
        model_id="bartowski/gemma-3-4b-it-GGUF",
        runtime=None,
    ),
    TransformerModelProfile(
        name="gemma3-12b",
        model_id="bartowski/gemma-3-12b-it-GGUF",
        runtime=None,
    ),
]

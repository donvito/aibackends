from aibackends.core.registry import TransformerModelProfile


MODEL_PROFILES = [
    TransformerModelProfile(
        name="llama3-8b",
        model_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
        runtime=None,
    ),
    TransformerModelProfile(
        name="mistral-7b",
        model_id="bartowski/Mistral-7B-Instruct-v0.3-GGUF",
        runtime=None,
    ),
    TransformerModelProfile(
        name="phi4-mini",
        model_id="bartowski/phi-4-mini-instruct-GGUF",
        runtime=None,
    ),
    TransformerModelProfile(
        name="minilm-l6",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        runtime=None,
    ),
    TransformerModelProfile(
        name="bge-small",
        model_id="BAAI/bge-small-en-v1.5",
        runtime=None,
    ),
    TransformerModelProfile(
        name="openai-privacy",
        model_id="openai/privacy-filter",
        runtime=None,
    ),
]

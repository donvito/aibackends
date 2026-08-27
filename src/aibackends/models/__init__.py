"""Built-in supported model refs and model profiles.

Each module in this package can expose `MODEL_PROFILES` or `MODEL_PROFILE`.
The profile registry discovers those files at runtime.
"""

from aibackends.core.model_registry import available_models, get_model_ref
from aibackends.core.registry import ModelRef

BGE_SMALL = ModelRef(name="bge-small")
GEMMA3_270M_IT = ModelRef(name="gemma3-270m-it")
GEMMA3_4B = ModelRef(name="gemma3-4b")
GEMMA3_12B = ModelRef(name="gemma3-12b")
GEMMA4_E2B = ModelRef(name="gemma4-e2b")
GEMMA4_E4B = ModelRef(name="gemma4-e4b")
LFM25_2_6B = ModelRef(name="lfm2.5-2.6b")
LFM25_VL_3B = ModelRef(name="lfm2.5-vl-3b")
LLAMA3_8B = ModelRef(name="llama3-8b")
MINILM_L6 = ModelRef(name="minilm-l6")
MISTRAL_7B = ModelRef(name="mistral-7b")
OPENAI_PRIVACY = ModelRef(name="openai-privacy")
PHI4_MINI = ModelRef(name="phi4-mini")
QWEN3_VL_4B = ModelRef(name="qwen3-vl-4b")
QWEN3_VL_8B = ModelRef(name="qwen3-vl-8b")
QWEN38_27B = ModelRef(name="qwen3.8-27b")

__all__ = [
    "available_models",
    "BGE_SMALL",
    "GEMMA3_270M_IT",
    "GEMMA3_4B",
    "GEMMA3_12B",
    "GEMMA4_E2B",
    "GEMMA4_E4B",
    "get_model_ref",
    "LFM25_2_6B",
    "LFM25_VL_3B",
    "LLAMA3_8B",
    "MINILM_L6",
    "MISTRAL_7B",
    "ModelRef",
    "OPENAI_PRIVACY",
    "PHI4_MINI",
    "QWEN3_VL_4B",
    "QWEN3_VL_8B",
    "QWEN38_27B",
]

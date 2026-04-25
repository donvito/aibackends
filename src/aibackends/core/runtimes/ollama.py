from aibackends.core.registry import RuntimeSpec
from aibackends.core.runtimes.base import OpenAICompatibleRuntime


class OllamaRuntime(OpenAICompatibleRuntime):
    provider_name = "ollama"
    default_base_url = "http://localhost:11434/v1"


RUNTIME_SPEC = RuntimeSpec(name="ollama", factory=OllamaRuntime)

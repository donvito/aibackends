from aibackends.core.registry import RuntimeSpec
from aibackends.core.runtimes.base import OpenAICompatibleRuntime


class GroqRuntime(OpenAICompatibleRuntime):
    provider_name = "groq"
    default_base_url = "https://api.groq.com/openai/v1"
    api_env_var = "GROQ_API_KEY"


RUNTIME_SPEC = RuntimeSpec(name="groq", factory=GroqRuntime)

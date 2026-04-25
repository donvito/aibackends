from aibackends.core.registry import RuntimeSpec
from aibackends.core.runtimes.base import OpenAICompatibleRuntime


class TogetherRuntime(OpenAICompatibleRuntime):
    provider_name = "together"
    default_base_url = "https://api.together.xyz/v1"
    api_env_var = "TOGETHER_API_KEY"


RUNTIME_SPEC = RuntimeSpec(name="together", factory=TogetherRuntime)

from aibackends.core.runtimes.base import OpenAICompatibleRuntime


class LMStudioRuntime(OpenAICompatibleRuntime):
    provider_name = "lmstudio"
    default_base_url = "http://localhost:1234/v1"

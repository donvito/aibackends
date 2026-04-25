from aibackends.core.registry import RuntimeSpec
from aibackends.core.runtimes.base import AnthropicMessagesRuntime


class AnthropicRuntime(AnthropicMessagesRuntime):
    """Anthropic Messages API runtime."""


RUNTIME_SPEC = RuntimeSpec(name="anthropic", factory=AnthropicRuntime)

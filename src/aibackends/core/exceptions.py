class AIBackendsError(Exception):
    """Base exception for AIBackends."""


class ConfigurationError(AIBackendsError):
    """Raised when configuration is missing or invalid."""


class RuntimeNotConfiguredError(ConfigurationError):
    """Raised when no runtime has been configured."""


class RuntimeImportError(AIBackendsError):
    """Raised when an optional runtime dependency is unavailable."""


class RuntimeRequestError(AIBackendsError):
    """Raised when a runtime request fails."""


class ModelResolutionError(AIBackendsError):
    """Raised when a model alias or path cannot be resolved."""


class TaskExecutionError(AIBackendsError):
    """Raised when a task cannot complete successfully."""


class ValidationRetryExhaustedError(TaskExecutionError):
    """Raised when structured output validation keeps failing."""


class WorkflowStepError(AIBackendsError):
    """Raised when a workflow step fails."""

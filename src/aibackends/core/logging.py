from __future__ import annotations

import logging
from typing import Any

from aibackends.core.types import RuntimeConfig, StepLog, TaskLog

LOGGER_NAME = "aibackends"
logger = logging.getLogger(LOGGER_NAME)
logger.addHandler(logging.NullHandler())


def configure_logging(level: int | str | None = None) -> logging.Logger:
    if level is not None:
        logger.setLevel(level)
    if not logger.handlers or all(
        isinstance(handler, logging.NullHandler) for handler in logger.handlers
    ):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        logger.handlers = [handler]
    return logger


def _format_metadata(metadata: dict[str, Any]) -> str:
    return " ".join(f"{key}={value}" for key, value in metadata.items())


def emit_task_log(log: TaskLog) -> None:
    configure_logging()
    suffix = _format_metadata(log.metadata)
    body = f"{log.task_name} {log.status}"
    if log.elapsed_ms is not None:
        body += f" | elapsed={log.elapsed_ms}ms"
    if suffix:
        body += f" {suffix}"
    logger.info(body)


def emit_step_log(log: StepLog, config: RuntimeConfig | None = None) -> None:
    configure_logging()
    suffix = _format_metadata(log.metadata)
    body = f"{log.task_name}:{log.step_name} {log.status}"
    if log.elapsed_ms is not None:
        body += f" | elapsed={log.elapsed_ms}ms"
    if suffix:
        body += f" {suffix}"
    logger.info(body)
    if config and config.on_step_complete:
        config.on_step_complete(log)

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel

from aibackends.core.config import get_runtime, parse_model_text, parse_runtime_text
from aibackends.core.model_manager import ModelManager
from aibackends.core.types import RuntimeConfig
from aibackends.tasks.registry import create_task, get_task

app = typer.Typer(help="AIBackends CLI")


@app.command("task")
def run_task(
    name: str = typer.Argument(..., help="Task name, e.g. extract-invoice or summarize."),
    input: str = typer.Option(..., "--input", help="Input path or raw text."),
    labels: str | None = typer.Option(
        None, help="Comma-separated labels for classify or GLiNER redact-pii."
    ),
    backend: str = typer.Option("gliner", help="PII backend to use."),
    device: str | None = typer.Option(None, help="Optional device override, e.g. cuda."),
    schema: str | None = typer.Option(None, help="Dotted import path for generic extract schema."),
    runtime: str | None = typer.Option(None, help="Runtime override."),
    model: str | None = typer.Option(None, help="Model override."),
) -> None:
    task = get_task(name)
    kwargs: dict[str, Any] = {}
    parsed_labels = _parse_labels(labels)
    if task.requires_labels and not parsed_labels:
        raise typer.BadParameter(f"--labels is required for {task.name}")
    if task.accepts_labels and parsed_labels is not None:
        kwargs["labels"] = parsed_labels
    if task.accepts_backend:
        kwargs["backend"] = backend
    if device is not None:
        kwargs["device"] = device
    if task.requires_schema:
        if not schema:
            raise typer.BadParameter(f"--schema is required for {task.name}")
        kwargs["schema"] = _load_schema(schema)
    task_config: dict[str, Any] = {}
    if task.accepts_runtime:
        task_config["runtime"] = parse_runtime_text(runtime)
    if task.accepts_model:
        task_config["model"] = parse_model_text(model)

    task_instance = create_task(task, **task_config)
    result = task_instance.run(input, **kwargs)
    typer.echo(_serialize(result))


@app.command("pull")
def pull_model(
    model: str = typer.Argument(..., help="Model alias, HF repo, or local file."),
    runtime: str = typer.Option(..., "--runtime", help="Runtime name."),
    cache_dir: str | None = typer.Option(
        None, "--cache-dir", help="Optional Hugging Face cache directory override."
    ),
) -> None:
    manager = ModelManager(cache_dir=cache_dir)
    location = manager.pull_model(
        RuntimeConfig.model_validate(
            {
                "runtime": parse_runtime_text(runtime),
                "model": parse_model_text(model),
                "cache_dir": cache_dir,
            }
        )
    )
    typer.echo(_serialize(location))


@app.command("check")
def check_runtime(
    runtime: str = typer.Argument(..., help="Runtime name."),
    model: str | None = typer.Option(None, "--model", help="Optional model to resolve."),
    base_url: str | None = typer.Option(None, "--base-url", help="Optional base URL override."),
) -> None:
    client = get_runtime(
        RuntimeConfig.model_validate(
            {
                "runtime": parse_runtime_text(runtime),
                "model": parse_model_text(model),
                "base_url": base_url,
            }
        )
    )
    typer.echo(
        _serialize(
            {
                "runtime": runtime,
                "client": client.__class__.__name__,
                "model": client.model_name,
                "base_url": base_url,
            }
        )
    )


def _load_schema(import_path: str) -> type[BaseModel]:
    module_name, _, attr = import_path.rpartition(".")
    if not module_name or not attr:
        raise typer.BadParameter(
            "Schema must be a dotted import path, e.g. package.module.SchemaName"
        )
    module = importlib.import_module(module_name)
    schema = getattr(module, attr)
    if not isinstance(schema, type) or not issubclass(schema, BaseModel):
        raise typer.BadParameter("Schema must resolve to a Pydantic model class.")
    return schema


def _parse_labels(value: str | None) -> list[str] | None:
    if value is None:
        return None
    labels = [item.strip() for item in value.split(",") if item.strip()]
    return labels or None


def _serialize(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if is_dataclass(value) and not isinstance(value, type):
        return json.dumps(asdict(value), indent=2, default=str)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, default=str)

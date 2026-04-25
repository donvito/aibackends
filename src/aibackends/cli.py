from __future__ import annotations

import importlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import typer
from pydantic import BaseModel

from aibackends.core.config import get_runtime
from aibackends.core.model_manager import ModelManager
from aibackends.core.types import RuntimeConfig
from aibackends.tasks import (
    analyse_sales_call,
    analyse_video_ad,
    classify,
    embed,
    extract,
    extract_invoice,
    redact_pii,
    summarize,
)

app = typer.Typer(help="AIBackends CLI")


@app.command("task")
def run_task(
    name: str = typer.Argument(..., help="Task name, e.g. extract-invoice or summarize."),
    input: str = typer.Option(..., "--input", help="Input path or raw text."),
    labels: str | None = typer.Option(
        None, help="Comma-separated labels for classify or GLiNER redact-pii."
    ),
    backend: str = typer.Option("gliner", help="PII backend to use."),
    schema: str | None = typer.Option(None, help="Dotted import path for generic extract schema."),
    runtime: str | None = typer.Option(None, help="Runtime override."),
    model: str | None = typer.Option(None, help="Model override."),
) -> None:
    normalized = name.replace("_", "-").lower()
    result: Any
    if normalized == "extract-invoice":
        result = extract_invoice(input, runtime=runtime, model=model)
    elif normalized == "redact-pii":
        pii_labels = [item.strip() for item in labels.split(",") if item.strip()] if labels else None
        result = redact_pii(input, backend=backend, labels=pii_labels)
    elif normalized == "classify":
        if not labels:
            raise typer.BadParameter("--labels is required for classify")
        result = classify(
            input, labels=[item.strip() for item in labels.split(",")], runtime=runtime, model=model
        )
    elif normalized == "summarize":
        result = summarize(input, runtime=runtime, model=model)
    elif normalized == "extract":
        if not schema:
            raise typer.BadParameter("--schema is required for extract")
        result = extract(input, schema=_load_schema(schema), runtime=runtime, model=model)
    elif normalized == "embed":
        result = embed(input, runtime=runtime, model=model)
    elif normalized == "analyse-sales-call":
        result = analyse_sales_call(input, runtime=runtime, model=model)
    elif normalized == "analyse-video-ad":
        result = analyse_video_ad(input, runtime=runtime, model=model)
    else:
        raise typer.BadParameter(f"Unknown task: {name}")
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
    location = manager.pull_model(RuntimeConfig(runtime=runtime, model=model, cache_dir=cache_dir))
    typer.echo(_serialize(location))


@app.command("check")
def check_runtime(
    runtime: str = typer.Argument(..., help="Runtime name."),
    model: str | None = typer.Option(None, "--model", help="Optional model to resolve."),
    base_url: str | None = typer.Option(None, "--base-url", help="Optional base URL override."),
) -> None:
    client = get_runtime(RuntimeConfig(runtime=runtime, model=model, base_url=base_url))
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

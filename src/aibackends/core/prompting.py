from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from aibackends.core.exceptions import ConfigurationError
from aibackends.core.types import Message, RuntimeConfig


@dataclass(slots=True)
class PromptRenderResult:
    prompt: str
    format_used: str
    template_source: str | None = None


def normalise_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item["text"]))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        return str(content.get("text", content))
    return str(content)


def merge_system_messages(messages: list[Message]) -> list[Message]:
    merged: list[Message] = []
    system_parts: list[str] = []
    index = 0

    while index < len(messages) and messages[index].get("role") == "system":
        content = normalise_message_content(messages[index].get("content", "")).strip()
        if content:
            system_parts.append(content)
        index += 1

    if system_parts:
        merged.append({"role": "system", "content": "\n\n".join(system_parts)})

    merged.extend(messages[index:])
    return merged


def render_messages_as_text(messages: list[Message]) -> str:
    rendered: list[str] = []
    for message in merge_system_messages(messages):
        role = message.get("role", "user").upper()
        rendered.append(f"{role}: {normalise_message_content(message.get('content', ''))}")
    return "\n\n".join(rendered)


def schema_prompt(schema: type[BaseModel]) -> str:
    json_schema = schema.model_json_schema()
    template = _json_template_from_schema(json_schema, json_schema.get("$defs", {}))
    return (
        "Return exactly one JSON object with extracted values. "
        "Do not include markdown fences, commentary, or prose. "
        "Replace the example values in the template with values from the input. "
        "Use null for missing optional values. "
        "Do not return JSON Schema keywords such as title, type, properties, items, anyOf, required, or $defs. "
        f"JSON template:\n{json.dumps(template, indent=2)}"
    )


def build_prompt_messages(
    messages: list[Message],
    *,
    schema: type[BaseModel] | None = None,
) -> list[Message]:
    rendered_messages = list(messages)
    if schema is not None:
        rendered_messages = [
            {"role": "system", "content": schema_prompt(schema)},
            *rendered_messages,
        ]
    return merge_system_messages(rendered_messages)


class PromptRenderer:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config

    def render(
        self,
        messages: list[Message],
        *,
        tokenizer: Any | None = None,
        schema: type[BaseModel] | None = None,
    ) -> PromptRenderResult:
        prompt_messages = build_prompt_messages(messages, schema=schema)
        if self.config.prompt_format == "text":
            return PromptRenderResult(
                prompt=render_messages_as_text(prompt_messages),
                format_used="text",
            )

        template_override, template_source = self._resolve_template_override()
        if template_override is not None:
            return self._render_with_template(
                tokenizer,
                prompt_messages,
                chat_template=template_override,
                template_source=template_source,
            )

        if getattr(tokenizer, "chat_template", None):
            return self._render_with_template(
                tokenizer,
                prompt_messages,
                template_source="tokenizer",
            )

        if self.config.prompt_format == "chat_template":
            raise ConfigurationError(
                "No chat template is available for this tokenizer. "
                "Provide `chat_template` or `chat_template_path`, or switch to `prompt_format=\"auto\"` or `prompt_format=\"text\"`."
            )

        return PromptRenderResult(
            prompt=render_messages_as_text(prompt_messages),
            format_used="text",
        )

    def _render_with_template(
        self,
        tokenizer: Any | None,
        messages: list[Message],
        *,
        chat_template: str | None = None,
        template_source: str | None = None,
    ) -> PromptRenderResult:
        if tokenizer is None or not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise ConfigurationError(
                "Prompt format `chat_template` requires a tokenizer with `apply_chat_template`."
            )

        apply_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if chat_template is not None:
            apply_kwargs["chat_template"] = chat_template

        try:
            prompt = str(tokenizer.apply_chat_template(messages, **apply_kwargs))
        except Exception as exc:
            detail = f" from {template_source}" if template_source else ""
            raise ConfigurationError(f"Failed to render chat template{detail}: {exc}") from exc

        return PromptRenderResult(
            prompt=prompt,
            format_used="chat_template",
            template_source=template_source,
        )

    def _resolve_template_override(self) -> tuple[str | None, str | None]:
        if self.config.chat_template is not None:
            return self.config.chat_template, "inline"
        if self.config.chat_template_path is None:
            return None, None

        template_path = Path(self.config.chat_template_path).expanduser()
        if not template_path.exists():
            raise ConfigurationError(f"Chat template file does not exist: {template_path}")
        if not template_path.is_file():
            raise ConfigurationError(f"Chat template path is not a file: {template_path}")
        return template_path.read_text(encoding="utf-8"), str(template_path)


def _json_template_from_schema(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    if "$ref" in schema:
        ref = str(schema["$ref"])
        if ref.startswith("#/$defs/"):
            return _json_template_from_schema(defs[ref.split("/")[-1]], defs)

    if "const" in schema:
        return schema["const"]

    if "enum" in schema and schema["enum"]:
        return schema["enum"][0]

    if "anyOf" in schema:
        variants = [variant for variant in schema["anyOf"] if variant.get("type") != "null"]
        if variants:
            return _json_template_from_schema(variants[0], defs)
        return None

    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        return {
            key: _json_template_from_schema(value, defs)
            for key, value in schema.get("properties", {}).items()
        }
    if schema_type == "array":
        item_schema = schema.get("items", {})
        return [_json_template_from_schema(item_schema, defs)]
    if schema_type == "string":
        return "..."
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False
    return None

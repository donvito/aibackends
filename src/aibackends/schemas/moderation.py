from __future__ import annotations

from typing import Literal, TypeAlias

from aibackends.core.types import AIBackendsModel

SafetyVerdict: TypeAlias = Literal["safe", "unsafe"]
RefusalVerdict: TypeAlias = Literal["refusal", "compliance"]
HarmCategory: TypeAlias = Literal[
    "violence_and_weapons",
    "non_violent_crime",
    "sexual_content",
    "hate_and_discrimination",
    "self_harm_and_suicide",
    "pii_exposure",
    "misinformation",
    "copyright_violation",
    "child_safety",
    "political_manipulation",
    "unethical_conduct",
    "regulated_advice",
    "privacy_violation",
    "other",
    "benign",
]
JailbreakStrategy: TypeAlias = Literal[
    "prompt_injection",
    "jailbreak_attempt",
    "policy_evasion",
    "instruction_override",
    "system_prompt_exfiltration",
    "data_exfiltration",
    "roleplay_bypass",
    "hypothetical_bypass",
    "obfuscated_attack",
    "multi_step_attack",
    "social_engineering",
    "benign",
]


class PromptModeration(AIBackendsModel):
    prompt: str
    is_safe: bool
    safety: SafetyVerdict
    toxicity: list[HarmCategory]
    jailbreak: list[JailbreakStrategy]
    backend_used: str
    model_id: str


class ResponseModeration(AIBackendsModel):
    response: str
    prompt: str | None = None
    is_safe: bool
    safety: SafetyVerdict
    toxicity: list[HarmCategory]
    refusal: RefusalVerdict
    backend_used: str
    model_id: str

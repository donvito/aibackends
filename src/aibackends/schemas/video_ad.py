from __future__ import annotations

from aibackends.core.types import AIBackendsModel


class VideoAdReport(AIBackendsModel):
    hook_strength: float
    key_messages: list[str]
    cta_clarity: float
    emotion_arc: list[str]

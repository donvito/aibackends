"""Qwen 3 VL 4B image understanding workflow.

Reads ``examples/data/images/receipt1.jpeg`` and asks Qwen 3 VL 4B to describe
the image and extract any visible text.

Requires one of:
    pip install 'aibackends[llamacpp]'
    pip install 'aibackends[llamacpp-metal]'
    pip install 'aibackends[llamacpp-cuda]'

Using ``model=QWEN3_VL_4B`` with ``runtime=LLAMACPP`` downloads the Qwen3-VL
GGUF and matching ``mmproj`` projector automatically.

Qwen3-VL support in ``llama-cpp-python`` is still moving quickly, so this
example expects a recent vision-capable build that includes Qwen VL chat
handler support.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import QWEN3_VL_4B
from aibackends.runtimes import LLAMACPP
from aibackends.steps.enrich import VisionExtractor
from aibackends.steps.ingest import ImageIngestor
from aibackends.workflows import Pipeline


class ImageUnderstandingResult(BaseModel):
    description: str
    visible_text: str | None = None


class QwenImageUnderstandingWorkflow(Pipeline):
    steps = [
        ImageIngestor(),
        VisionExtractor(
            schema=ImageUnderstandingResult,
            prompt=(
                "Describe the image. If there is readable text, copy it into "
                "visible_text. Keep the answer concise."
            ),
        ),
    ]


def main() -> None:
    try:
        image_path = Path(__file__).parent.parent / "data" / "images" / "receipt1.jpeg"
        workflow = QwenImageUnderstandingWorkflow(
            runtime=LLAMACPP,
            model=QWEN3_VL_4B,
        )
        result = workflow.run(image_path)
        print(result.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

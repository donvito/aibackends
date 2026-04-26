"""Gemma 4 image understanding workflow.

Reads ``examples/data/images/receipt1.jpeg`` and asks Gemma 4 to describe the
image and extract any visible text.

Requires one of:
    pip install 'aibackends[llamacpp]'
    pip install 'aibackends[llamacpp-metal]'
    pip install 'aibackends[llamacpp-cuda]'

Using ``model=GEMMA4_E2B`` with ``runtime=LLAMACPP`` downloads the Gemma 4 GGUF
and matching ``mmproj`` projector automatically.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pydantic import BaseModel

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.steps.enrich import VisionExtractor
from aibackends.steps.ingest import ImageIngestor
from aibackends.workflows import Pipeline


class ImageUnderstandingResult(BaseModel):
    description: str
    visible_text: str | None = None


class GemmaImageUnderstandingWorkflow(Pipeline):
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
        workflow = GemmaImageUnderstandingWorkflow(
            runtime=LLAMACPP,
            model=GEMMA4_E2B,
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

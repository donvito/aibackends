"""Qwen 3 VL 4B receipt extraction workflow.

Reads ``examples/data/images/receipt1.jpeg`` and extracts structured receipt
data with Qwen 3 VL 4B running through the llama.cpp runtime.

The script prints structured JSON with the source image path and extracted
receipt fields.

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

from pydantic import BaseModel, Field

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import QWEN3_VL_4B
from aibackends.runtimes import LLAMACPP
from aibackends.schemas.common import LineItem
from aibackends.steps.enrich import VisionExtractor
from aibackends.steps.ingest import ImageIngestor
from aibackends.workflows import Pipeline


class ReceiptResult(BaseModel):
    merchant: str | None = None
    receipt_date: str | None = None
    receipt_time: str | None = None
    receipt_number: str | None = None
    currency: str | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    payment_method: str | None = None
    line_items: list[LineItem] = Field(default_factory=list)


class ReceiptExampleOutput(BaseModel):
    source: str
    receipt: ReceiptResult


class QwenReceiptWorkflow(Pipeline):
    steps = [
        ImageIngestor(),
        VisionExtractor(
            schema=ReceiptResult,
            prompt=(
                "Extract structured data from this receipt. "
                "Return JSON with these fields: merchant, receipt_date, "
                "receipt_time, receipt_number, currency, subtotal, tax, total, "
                "payment_method, and line_items. "
                "Use null for missing scalar fields and [] for missing line_items. "
                "Extract monetary values as numbers, not strings. "
                "Do not summarize or describe the image."
            ),
        ),
    ]


def main() -> None:
    try:
        image_path = Path(__file__).parent.parent / "data" / "images" / "receipt2.jpeg"
        workflow = QwenReceiptWorkflow(
            runtime=LLAMACPP,
            model=QWEN3_VL_4B,
        )
        receipt = workflow.run(image_path)
        result = ReceiptExampleOutput(source=str(image_path), receipt=receipt)
        print(result.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

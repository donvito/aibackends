"""Extract PII and contract clauses from a complete document with global offsets.

Run from the repository root:
    python3 -m examples.gliner25.long_context --model small --device cpu

Requires:
    pip install -e '.[gliner2]'
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import add_model_arguments, assert_source_spans, load_extractor, print_result

CONTRACT_PATH = Path(__file__).parents[1] / "data" / "contract.txt"
ENTITY_TYPES = {
    "person": "Names of people who are parties to the agreement",
    "email": "Email addresses",
    "phone_number": "Telephone numbers",
    "postal_address": "Complete mailing or property addresses",
    "bank_account": "Bank or payment account identifiers",
    "obligation": "Complete clauses describing a party's required action",
    "termination_clause": "Complete clauses describing how the agreement can end",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Words per chunk; deliberately small so the sample contract spans chunks.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=32,
        help="Overlapping words between adjacent chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size < 2:
        raise SystemExit("--chunk-size must be at least 2")
    if not 0 <= args.chunk_overlap < args.chunk_size:
        raise SystemExit("--chunk-overlap must be non-negative and smaller than --chunk-size")

    text = CONTRACT_PATH.read_text(encoding="utf-8")
    started = time.perf_counter()
    extractor, model_id, device = load_extractor(args.model, args.device)
    load_seconds = time.perf_counter() - started

    started = time.perf_counter()
    result = extractor.extract_entities_long(
        text,
        ENTITY_TYPES,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        include_spans=True,
        include_confidence=True,
    )
    elapsed_seconds = time.perf_counter() - started
    span_count = assert_source_spans(text, result)

    print(f"model: {model_id}")
    print(f"device: {device}")
    print(f"load: {load_seconds:.2f}s")
    print(
        f"long extraction: {elapsed_seconds:.2f}s across {len(text.split())} words; "
        f"{span_count} source-verified spans"
    )
    print_result(result)


if __name__ == "__main__":
    main()

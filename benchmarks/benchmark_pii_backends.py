"""Benchmark PII backend model caching: cold load vs warm redaction calls.

Measures, for a PII backend (GLiNER by default):

1. ``backend.load()`` — the one-time model load (cold).
2. Warm ``backend.redact(...)`` calls that reuse the cached model.
3. A redact call that includes the load, for the total first-call cost.

Writes a markdown report to ``benchmarks/reports/`` for committing to the repo.

Usage:
    python benchmarks/benchmark_pii_backends.py --backend gliner --warm-calls 5

Requires:
    pip install 'aibackends[pii]'
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from _reporting import (
    TimingStats,
    benchmark_lock,
    environment_lines,
    stats_table,
    write_report,
)

from aibackends.backends.pii import get_pii_backend
from aibackends.core.exceptions import AIBackendsError

SAMPLE_TEXT = (
    "Hi support, my username is johndoe88 and I cannot log in. "
    "Reach me at johnd@example.com or (555) 123-4567. "
    "My colleague jane.roe@example.org can confirm the issue."
)


def _clear_backend_cache(backend_name: str) -> None:
    if backend_name == "gliner":
        from aibackends.backends.pii.gliner import clear_model_cache

        clear_model_cache()
    elif backend_name in {"openai-privacy", "openai_privacy"}:
        from aibackends.backends.pii.openai_privacy import clear_pipeline_cache

        clear_pipeline_cache()


def run_benchmark(args: argparse.Namespace) -> list[str]:
    backend = get_pii_backend(args.backend)

    cold_load = TimingStats("`backend.load()` (cold)")
    first_redact = TimingStats("First `redact()` including load")
    warm_redacts = TimingStats("Warm `redact()` calls")

    print(f"Benchmarking PII backend {backend.name}", flush=True)

    print("1/2 redact including model load...", flush=True)
    _clear_backend_cache(backend.name)
    started = time.perf_counter()
    backend.redact(SAMPLE_TEXT)
    first_redact.add((time.perf_counter() - started) * 1000)

    print(f"2/2 explicit load, then {args.warm_calls} warm redacts...", flush=True)
    _clear_backend_cache(backend.name)
    started = time.perf_counter()
    backend.load()
    cold_load.add((time.perf_counter() - started) * 1000)
    for _ in range(args.warm_calls):
        started = time.perf_counter()
        backend.redact(SAMPLE_TEXT)
        warm_redacts.add((time.perf_counter() - started) * 1000)

    speedup = first_redact.mean_ms / warm_redacts.mean_ms if warm_redacts.samples_ms else 0.0

    lines = [
        "# PII Backend Benchmark",
        "",
        f"Backend `{backend.name}` (model `{backend.model_id}`), "
        f"sample text of {len(SAMPLE_TEXT)} characters.",
        "",
        "## Environment",
        "",
        *environment_lines(("gliner", "transformers", "torch")),
        "",
        "## Results",
        "",
        *stats_table([first_redact, cold_load, warm_redacts]),
        "",
        f"Warm redactions are **{speedup:,.1f}x** faster than the first call "
        "that includes the model load.",
        "",
        "## Notes",
        "",
        "- The model is cached process-wide after the first call; use",
        "  `backend.load()` at startup to move the load cost out of the first",
        "  redaction.",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="gliner")
    parser.add_argument("--warm-calls", type=int, default=10, help="Warm samples per scenario.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    try:
        with benchmark_lock():
            lines = run_benchmark(args)
    except AIBackendsError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    report_path = write_report(
        name=f"pii-{args.backend}",
        lines=lines,
        output_dir=args.output_dir,
    )
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

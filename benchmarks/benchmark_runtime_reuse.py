"""Benchmark runtime reuse: model load per call vs the process-wide runtime cache.

Measures three scenarios against the same runtime and model:

1. ``reuse_runtime=False`` — a fresh runtime per call, so every call pays the
   full model load (the pre-0.3 behavior).
2. ``reuse_runtime=True`` — the first call loads the model, subsequent calls
   reuse the cached runtime and skip the load.
3. ``preload()`` — the load cost is paid up front, so even the first request
   is warm.

Only models from the recommended ``aibackends.models`` catalog are accepted,
so committed reports stay comparable across runs.

Writes a markdown report to ``benchmarks/reports/`` for committing to the repo.

Sample counts are configurable via ``--cold-calls`` and ``--warm-calls``
(default 10). With 10+ warm samples the report includes consistency
statistics (stdev, CV, p50/p95, drift) and per-segment trends that reveal
inference degradation over a sustained run; raise ``--warm-calls`` for a
longer soak.

Usage:
    python benchmarks/benchmark_runtime_reuse.py --runtime transformers \
        --model gemma3-270m-it --warm-calls 10

Requires:
    pip install 'aibackends[transformers]'  # or [llamacpp]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

# Keep benchmark output readable; transformers is imported lazily by the runtime.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from _reporting import (
    TimingStats,
    benchmark_lock,
    consistency_table,
    ensure_gpu_dlls,
    environment_lines,
    segment_trend_lines,
    stats_table,
    write_report,
)

from aibackends.core.config import clear_runtime_cache, get_runtime
from aibackends.core.exceptions import AIBackendsError
from aibackends.models import available_models

PROMPT = "Reply with a short greeting."


def _resolve_catalog_model(name: str, runtime: str) -> str:
    catalog = available_models(runtime=runtime)
    if name not in catalog:
        supported = ", ".join(sorted(catalog)) or "none"
        raise SystemExit(
            f"Model '{name}' is not in the recommended catalog for runtime "
            f"'{runtime}'. Supported models: {supported}"
        )
    return name


def _time_call(overrides: dict[str, Any]) -> float:
    started = time.perf_counter()
    runtime = get_runtime(overrides)
    runtime.complete([{"role": "user", "content": PROMPT}])
    return (time.perf_counter() - started) * 1000


DEFAULT_MODEL = {"transformers": "gemma3-270m-it", "llamacpp": "gemma4-e2b"}


def run_benchmark(args: argparse.Namespace) -> list[str]:
    if args.model is None:
        args.model = DEFAULT_MODEL.get(args.runtime, "gemma4-e2b")
    model = _resolve_catalog_model(args.model, args.runtime)
    overrides: dict[str, Any] = {
        "runtime": args.runtime,
        "model": model,
        "max_tokens": args.max_tokens,
    }

    no_reuse = TimingStats("Fresh runtime per call (`reuse_runtime=False`)")
    cold_call = TimingStats("First call with reuse (load + inference)")
    warm_calls = TimingStats("Warm calls with reuse")
    preload_cost = TimingStats("`preload()` (load only)")
    post_preload = TimingStats("First call after `preload()`")

    print(f"Benchmarking {args.runtime} / {args.model}", flush=True)

    print(f"1/3 fresh runtime per call x{args.cold_calls}...", flush=True)
    clear_runtime_cache()
    for _ in range(args.cold_calls):
        no_reuse.add(_time_call({**overrides, "reuse_runtime": False}))

    print(f"2/3 reuse: 1 cold + {args.warm_calls} warm calls...", flush=True)
    clear_runtime_cache()
    cold_call.add(_time_call(overrides))
    for _ in range(args.warm_calls):
        warm_calls.add(_time_call(overrides))

    print("3/3 preload, then first call...", flush=True)
    clear_runtime_cache()
    started = time.perf_counter()
    runtime = get_runtime(overrides)
    runtime.preload()
    preload_cost.add((time.perf_counter() - started) * 1000)
    post_preload.add(_time_call(overrides))

    speedup = no_reuse.mean_ms / warm_calls.mean_ms if warm_calls.samples_ms else 0.0

    lines = [
        "# Runtime Reuse Benchmark",
        "",
        f"Runtime `{args.runtime}`, model `{args.model}`, "
        f"max_tokens {args.max_tokens}, prompt: {PROMPT!r}",
        "",
        "## Environment",
        "",
        *environment_lines(("transformers", "torch", "llama-cpp-python")),
        "",
        "## Results",
        "",
        *stats_table([no_reuse, cold_call, warm_calls, preload_cost, post_preload]),
        "",
        f"Warm calls are **{speedup:,.1f}x** faster than fresh-runtime calls "
        "(mean over mean).",
        "",
        "## Consistency",
        "",
        *consistency_table([warm_calls]),
        "",
        *segment_trend_lines(warm_calls),
        "",
        "## Notes",
        "",
        "- Fresh-runtime calls still benefit from the OS file cache after the",
        "  first load, so a true cold process start is slower than shown.",
        "- Warm calls measure inference only; the model load cost is paid once",
        "  per process (or explicitly via `preload()`).",
    ]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="transformers")
    parser.add_argument(
        "--model",
        default=None,
        help="Catalog model. Defaults to the smallest chat model for the runtime.",
    )
    parser.add_argument("--cold-calls", type=int, default=3, help="Fresh-runtime samples.")
    parser.add_argument("--warm-calls", type=int, default=10, help="Warm samples per scenario.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    ensure_gpu_dlls()
    try:
        with benchmark_lock():
            lines = run_benchmark(args)
    except AIBackendsError as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    report_path = write_report(
        name=f"runtime-reuse-{args.runtime}-{args.model}",
        lines=lines,
        output_dir=args.output_dir,
    )
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

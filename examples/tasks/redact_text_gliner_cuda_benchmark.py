import statistics
import sys
from time import perf_counter

from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import RedactPIITask, create_task
from _gliner_benchmark import (
    GLINER_LABELS,
    REPEAT_BLOCKS,
    build_large_benchmark_text,
)

RUNS = 5


def _format_duration(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def main() -> None:
    try:
        task = create_task(
            RedactPIITask,
            backend="gliner",
            labels=GLINER_LABELS,
            device="cuda",
        )
        source_path, benchmark_text = build_large_benchmark_text()

        durations: list[float] = []
        last_result = None
        for _ in range(RUNS):
            started = perf_counter()
            last_result = task.run(benchmark_text)
            durations.append(perf_counter() - started)

        assert last_result is not None

        cold_run = durations[0]
        warm_runs = durations[1:]

        print(f"Input source: {source_path}")
        print(f"Repeated copies: {REPEAT_BLOCKS}")
        print(f"Input chars: {len(benchmark_text)}")
        print("Device: cuda")
        print(f"Runs: {RUNS}")
        print(f"Cold run: {_format_duration(cold_run)}")
        if warm_runs:
            print(f"Warm avg: {_format_duration(statistics.mean(warm_runs))}")
            print(f"Warm min: {_format_duration(min(warm_runs))}")
            print(f"Speedup vs cold: {cold_run / statistics.mean(warm_runs):.2f}x")
        print()
        print(
            "This benchmark uses a larger repeated contract sample and requires a CUDA-capable "
            "PyTorch and GLiNER setup. Compare it against "
            "redact_text_gliner_cache_benchmark.py to measure CPU vs CUDA."
        )
        print("Redacted preview:")
        print(last_result.redacted_text[:300])
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

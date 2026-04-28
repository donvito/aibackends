import statistics
import sys
from time import perf_counter

from _gliner_benchmark import GLINER_LABELS, REPEAT_BLOCKS, build_large_benchmark_text
from aibackends.core.exceptions import AIBackendsError
from aibackends.tasks import RedactPIITask, create_task

RUNS = 100
MAIN_BASELINE_COLD_MS = 13574.5
MAIN_BASELINE_WARM_MS = 14023.7


def _format_duration(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.1f} s"


def main() -> None:
    try:
        task = create_task(
            RedactPIITask,
            backend="gliner",
            labels=GLINER_LABELS,
        )
        source_path, benchmark_text = build_large_benchmark_text()

        durations: list[float] = []
        for _ in range(RUNS):
            started = perf_counter()
            task.run(benchmark_text)
            durations.append(perf_counter() - started)

        cold_run = durations[0]
        warm_runs = durations[1:]
        total_elapsed = sum(durations)

        main_estimated_total = (
            MAIN_BASELINE_COLD_MS / 1000
            + ((RUNS - 1) * MAIN_BASELINE_WARM_MS / 1000)
        )
        saved_seconds = main_estimated_total - total_elapsed

        print(f"Input source: {source_path}")
        print(f"Repeated copies: {REPEAT_BLOCKS}")
        print(f"Input chars: {len(benchmark_text)}")
        print(f"Runs: {RUNS}")
        print(f"Cold run: {_format_duration(cold_run)}")
        print(f"Warm avg: {_format_duration(statistics.mean(warm_runs))}")
        print(f"Warm min: {_format_duration(min(warm_runs))}")
        print(f"Actual 100-call total: {_format_seconds(total_elapsed)}")
        print()
        print("Committed main baseline used for comparison:")
        print(f"Main cold: {MAIN_BASELINE_COLD_MS:.1f} ms")
        print(f"Main warm avg: {MAIN_BASELINE_WARM_MS:.1f} ms")
        print(f"Estimated main 100-call total: {_format_seconds(main_estimated_total)}")
        print(f"Estimated time saved over 100 calls: {_format_seconds(saved_seconds)}")
        if saved_seconds > 0:
            print(
                f"Percent faster than main over 100 calls: "
                f"{(saved_seconds / main_estimated_total) * 100:.1f}%"
            )
        else:
            print(
                f"Percent slower than main over 100 calls: "
                f"{(-saved_seconds / main_estimated_total) * 100:.1f}%"
            )
        print()
        print(
            "These estimates use the committed-main benchmark measured earlier on the same machine "
            "and larger repeated contract sample."
        )
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

"""Run all benchmarks sequentially, one process at a time.

Each benchmark runs as its own subprocess so the previous model's RAM/VRAM is
fully released before the next benchmark starts — important on machines with
a single GPU. A shared lock file additionally prevents two benchmarks from
running concurrently.

Usage:
    python benchmarks/run_all.py --runtime transformers --warm-calls 10
    python benchmarks/run_all.py --skip pii --warm-calls 100
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BENCHMARKS_DIR = Path(__file__).parent

BENCHMARKS: dict[str, str] = {
    "runtime-reuse": "benchmark_runtime_reuse.py",
    "tasks": "benchmark_tasks.py",
    "pii": "benchmark_pii_backends.py",
}


def _build_command(name: str, script: str, args: argparse.Namespace) -> list[str]:
    command = [sys.executable, str(BENCHMARKS_DIR / script)]
    command += ["--warm-calls", str(args.warm_calls)]
    if name in {"runtime-reuse", "tasks"}:
        command += ["--runtime", args.runtime]
    if args.output_dir is not None:
        command += ["--output-dir", str(args.output_dir)]
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", default="transformers")
    parser.add_argument(
        "--warm-calls",
        type=int,
        default=10,
        help="Samples per scenario, forwarded to every benchmark.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=sorted(BENCHMARKS),
        help="Benchmarks to skip.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    results: dict[str, int] = {}
    for name, script in BENCHMARKS.items():
        if name in args.skip:
            print(f"=== {name}: skipped ===", flush=True)
            continue
        print(f"=== {name} ===", flush=True)
        completed = subprocess.run(_build_command(name, script, args), check=False)
        results[name] = completed.returncode
        status = "ok" if completed.returncode == 0 else f"failed ({completed.returncode})"
        print(f"=== {name}: {status} ===\n", flush=True)

    failed = {name: code for name, code in results.items() if code != 0}
    if failed:
        failures = ", ".join(sorted(failed))
        print(f"Completed with failures: {failures}", file=sys.stderr)
        raise SystemExit(1)
    print("All benchmarks completed.", flush=True)


if __name__ == "__main__":
    main()

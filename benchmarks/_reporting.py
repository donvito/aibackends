"""Shared helpers for benchmark scripts: timing stats, markdown reports, locking."""

from __future__ import annotations

import os
import platform
import statistics
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

REPORTS_DIR = Path(__file__).parent / "reports"
LOCK_FILE = Path(__file__).parent / ".benchmark.lock"


def ensure_gpu_dlls() -> None:
    """Import torch before llama_cpp so its bundled CUDA runtime DLLs are loadable.

    On Windows the prebuilt CUDA llama.cpp wheel links against CUDA runtime
    libraries that ship inside the torch package rather than system-wide.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        pass


@contextmanager
def benchmark_lock() -> Iterator[None]:
    """Serialize benchmark runs: models share one GPU, so only one may run at a time."""
    try:
        descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(
            f"Another benchmark appears to be running ({LOCK_FILE} exists). "
            "Benchmarks share one GPU, so run them one at a time. "
            "Delete the lock file if it is stale."
        ) from None
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.close(descriptor)
        yield
    finally:
        LOCK_FILE.unlink(missing_ok=True)


@dataclass
class TimingStats:
    label: str
    samples_ms: list[float] = field(default_factory=list)

    def add(self, elapsed_ms: float) -> None:
        self.samples_ms.append(elapsed_ms)

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    @property
    def stdev_ms(self) -> float:
        if len(self.samples_ms) < 2:
            return 0.0
        return statistics.stdev(self.samples_ms)

    def percentile_ms(self, fraction: float) -> float:
        ordered = sorted(self.samples_ms)
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    @property
    def drift_percent(self) -> float:
        """Mean of the last quarter of samples vs the first quarter, as a percentage.

        Positive values mean calls got slower over the run (possible degradation);
        values near zero mean stable inference times.
        """
        quarter = max(1, len(self.samples_ms) // 4)
        first = statistics.fmean(self.samples_ms[:quarter])
        last = statistics.fmean(self.samples_ms[-quarter:])
        if first == 0:
            return 0.0
        return (last - first) / first * 100

    def segment_means_ms(self, segments: int = 10) -> list[float]:
        count = len(self.samples_ms)
        segments = min(segments, count)
        size = count / segments
        means = []
        for index in range(segments):
            chunk = self.samples_ms[round(index * size) : round((index + 1) * size)]
            means.append(statistics.fmean(chunk))
        return means

    def row(self) -> str:
        return (
            f"| {self.label} | {len(self.samples_ms)} | {self.mean_ms:,.1f} "
            f"| {self.min_ms:,.1f} | {self.max_ms:,.1f} |"
        )

    def consistency_row(self) -> str:
        cv = (self.stdev_ms / self.mean_ms * 100) if self.mean_ms else 0.0
        return (
            f"| {self.label} | {self.stdev_ms:,.1f} | {cv:,.1f}% "
            f"| {self.percentile_ms(0.5):,.1f} | {self.percentile_ms(0.95):,.1f} "
            f"| {self.drift_percent:+,.1f}% |"
        )


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def _gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "none detected"
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ", ".join(names) if result.returncode == 0 and names else "none detected"


def environment_lines(extra_packages: tuple[str, ...] = ()) -> list[str]:
    lines = [
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Platform: {platform.platform()}",
        f"- Processor: {platform.processor() or 'unknown'}",
        f"- GPU: {_gpu_name()}",
        f"- Python: {platform.python_version()}",
        f"- aibackends: {_package_version('aibackends')}",
    ]
    for package in extra_packages:
        lines.append(f"- {package}: {_package_version(package)}")
    return lines


def stats_table(stats: list[TimingStats]) -> list[str]:
    lines = [
        "| Scenario | Samples | Mean (ms) | Min (ms) | Max (ms) |",
        "|---|---|---|---|---|",
    ]
    lines.extend(item.row() for item in stats)
    return lines


def consistency_table(stats: list[TimingStats]) -> list[str]:
    """Spread and drift for scenarios with enough samples to be meaningful."""
    eligible = [item for item in stats if len(item.samples_ms) >= 10]
    if not eligible:
        return []
    lines = [
        "| Scenario | Stdev (ms) | CV | p50 (ms) | p95 (ms) | Drift |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(item.consistency_row() for item in eligible)
    lines.extend(
        [
            "",
            "CV is the coefficient of variation (stdev / mean). Drift compares "
            "the mean of the last quarter of calls against the first quarter; "
            "positive drift means calls slowed down over the run.",
        ]
    )
    return lines


def segment_trend_lines(stats: TimingStats, segments: int = 10) -> list[str]:
    """Render per-segment means so degradation over a long run is visible."""
    if len(stats.samples_ms) < segments:
        return []
    means = stats.segment_means_ms(segments)
    header = " | ".join(f"S{index + 1}" for index in range(len(means)))
    values = " | ".join(f"{value:,.0f}" for value in means)
    return [
        f"Per-segment mean (ms) across the run, {stats.label}:",
        "",
        f"| {header} |",
        f"|{'---|' * len(means)}",
        f"| {values} |",
    ]


def slugify(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "-" for char in value.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-") or "unknown"


def write_report(*, name: str, lines: list[str], output_dir: Path | None = None) -> Path:
    directory = output_dir or REPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
    path = directory / f"{date_prefix}_{slugify(name)}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

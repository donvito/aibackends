"""Benchmark GLiNER PII redaction: cold first call vs warm subsequent calls.

The first ``backend.redact(...)`` call triggers the GLiNER model load (and a
Hugging Face download if the model is not cached yet). Later calls reuse the
in-process model cache, so their response times reflect pure inference.

Run with: uv run python benchmark_gliner_pii.py
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from aibackends.backends.pii import get_pii_backend

PII_LABELS = ["person_name", "email", "phone_number", "address"]
ROUNDS = 4


def main() -> None:
    data_dir = Path(__file__).parent / "examples" / "data" / "batch"
    paths = sorted(data_dir.glob("pii_note_*.txt"))
    docs = [(path.name, path.read_text(encoding="utf-8")) for path in paths]
    print(f"Documents: {len(docs)} | Rounds: {ROUNDS} | Labels: {', '.join(PII_LABELS)}")

    backend = get_pii_backend("gliner")

    rows: list[tuple[int, int, str, int, float]] = []
    call_no = 0
    bench_started = time.perf_counter()
    for round_no in range(1, ROUNDS + 1):
        for name, text in docs:
            call_no += 1
            started = time.perf_counter()
            result = backend.redact(text, labels=PII_LABELS)
            elapsed_ms = (time.perf_counter() - started) * 1000
            rows.append((call_no, round_no, name, len(result.entities_found), elapsed_ms))
    total_s = time.perf_counter() - bench_started

    print("\nPer-call response times:")
    print(f"{'call':>4}  {'round':>5}  {'document':<16}  {'entities':>8}  {'time_ms':>10}  note")
    for call, rnd, name, entities, ms in rows:
        note = "cold (includes model load)" if call == 1 else "warm"
        print(f"{call:>4}  {rnd:>5}  {name:<16}  {entities:>8}  {ms:>10.1f}  {note}")

    cold_ms = rows[0][4]
    warm_times = [ms for _, _, _, _, ms in rows[1:]]
    print("\nSummary:")
    print(f"  Documents processed : {len(rows)} calls ({len(docs)} unique docs x {ROUNDS} rounds)")
    print(f"  Total turnaround    : {total_s:.2f} s")
    print(f"  Throughput          : {len(rows) / total_s:.2f} docs/s (including cold load)")
    print(f"  Cold call (1st)     : {cold_ms:.1f} ms")
    print(f"  Warm calls (rest)   : avg {statistics.mean(warm_times):.1f} ms | "
          f"min {min(warm_times):.1f} ms | max {max(warm_times):.1f} ms")
    warm_total_s = sum(warm_times) / 1000
    print(f"  Warm throughput     : {len(warm_times) / warm_total_s:.2f} docs/s")
    print(f"  Speedup after load  : {cold_ms / statistics.mean(warm_times):.1f}x faster")


if __name__ == "__main__":
    main()

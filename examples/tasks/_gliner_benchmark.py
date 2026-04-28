from pathlib import Path

GLINER_LABELS = [
    "name",
    "email",
    "phone_number",
    "address",
    "idenfication_number",
    "passport_number",
    "account_number",
]
REPEAT_BLOCKS = 8


def benchmark_source_path() -> Path:
    return Path(__file__).parent.parent / "data" / "contract.txt"


def build_large_benchmark_text() -> tuple[Path, str]:
    source_path = benchmark_source_path()
    base_text = source_path.read_text(encoding="utf-8").strip()
    sections = [
        f"--- Contract copy {index} ---\n{base_text}"
        for index in range(1, REPEAT_BLOCKS + 1)
    ]
    return source_path, "\n\n".join(sections)

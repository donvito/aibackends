"""Zero-shot accuracy eval for the GLiNER2.5 extraction backend.

Runs the small / base / multi checkpoints against public datasets with no
fine-tuning and no in-context examples:

- ``fancyzhx/ag_news`` (test): 4-way topic classification via `classify_text`.
  Scored with accuracy and macro-F1.
- ``cornell-movie-review-data/rotten_tomatoes`` (test): binary sentiment via
  `classify_text`. Scored with accuracy and macro-F1.
- ``mneb/cross-ner`` politics (test): NER via `extract_entities`, using the
  label descriptions shipped with the dataset. Scored with micro precision /
  recall / F1 on exact (start, end, label) span matches.

Writes a markdown report to ``evals/reports/`` for committing to the repo.

Usage:
    python evals/eval_gliner25.py --models small base multi --samples 150

Requires:
    pip install 'aibackends[extraction]' datasets
"""

from __future__ import annotations

import argparse
import os
import platform
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

# Keep dataset/model download logging readable.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from aibackends.backends.extraction import get_extraction_backend
from aibackends.core.exceptions import AIBackendsError

REPORTS_DIR = Path(__file__).parent / "reports"
SEED = 20260825
NER_THRESHOLD = 0.5

# Dataset class names verbalized for zero-shot classification.
AG_NEWS_LABELS = {
    "World": "world news",
    "Sports": "sports",
    "Business": "business",
    "Sci/Tech": "science and technology",
}
ROTTEN_LABELS = {"neg": "negative", "pos": "positive"}


@dataclass
class ClassificationScore:
    dataset: str
    model: str
    accuracy: float
    macro_f1: float
    samples: int
    seconds: float


@dataclass
class NerScore:
    dataset: str
    model: str
    precision: float
    recall: float
    micro_f1: float
    gold_spans: int
    predicted_spans: int
    samples: int
    seconds: float


def _load_classification(
    dataset_id: str, label_map: dict[str, str], samples: int
) -> tuple[list[str], list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, split="test").shuffle(seed=SEED)
    names = dataset.features["label"].names
    texts: list[str] = []
    gold: list[str] = []
    for row in dataset.select(range(min(samples, len(dataset)))):
        texts.append(row["text"])
        gold.append(label_map[names[row["label"]]])
    return texts, gold


def _macro_f1(gold: list[str], predicted: list[str], labels: list[str]) -> float:
    scores = []
    for label in labels:
        true_positive = sum(
            1 for g, p in zip(gold, predicted, strict=True) if g == label and p == label
        )
        predicted_count = sum(1 for p in predicted if p == label)
        gold_count = sum(1 for g in gold if g == label)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / gold_count if gold_count else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def eval_classification(
    backend_name: str,
    model: str,
    dataset_name: str,
    dataset_id: str,
    label_map: dict[str, str],
    samples: int,
    batch_size: int,
) -> ClassificationScore:
    backend = get_extraction_backend(backend_name)
    texts, gold = _load_classification(dataset_id, label_map, samples)
    labels = list(label_map.values())

    started = time.perf_counter()
    results = backend.classify_text_batch(
        texts,
        tasks={"label": {"labels": labels}},
        model=model,
        batch_size=batch_size,
    )
    seconds = time.perf_counter() - started

    predicted = [result.value("label") or "" for result in results]
    hits = sum(1 for g, p in zip(gold, predicted, strict=True) if g == p)
    return ClassificationScore(
        dataset=dataset_name,
        model=model,
        accuracy=hits / len(gold),
        macro_f1=_macro_f1(gold, predicted, labels),
        samples=len(gold),
        seconds=seconds,
    )


def _load_ner(
    dataset_id: str, config: str, samples: int
) -> tuple[list[str], list[set[tuple[int, int, str]]], dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_id, config, split="test").shuffle(seed=SEED)
    label_descriptions: dict[str, str] = {
        entry["label"]: entry["description"] for entry in dataset[0]["schema"]["entities"]
    }
    texts: list[str] = []
    gold: list[set[tuple[int, int, str]]] = []
    for row in dataset.select(range(min(samples, len(dataset)))):
        texts.append(row["input"])
        spans = {
            (span["start"], span["end"], label)
            for label, entries in (row["output"].get("entities") or {}).items()
            for span in entries
        }
        gold.append(spans)
    return texts, gold, label_descriptions


def eval_ner(
    backend_name: str,
    model: str,
    dataset_name: str,
    dataset_id: str,
    config: str,
    samples: int,
    batch_size: int,
) -> NerScore:
    backend = get_extraction_backend(backend_name)
    texts, gold, label_descriptions = _load_ner(dataset_id, config, samples)

    started = time.perf_counter()
    results = backend.extract_entities_batch(
        texts,
        labels=label_descriptions,
        model=model,
        threshold=NER_THRESHOLD,
        batch_size=batch_size,
    )
    seconds = time.perf_counter() - started

    counts: Counter[str] = Counter()
    for gold_spans, result in zip(gold, results, strict=True):
        predicted_spans = {
            (entity.start, entity.end, entity.label)
            for entity in result.entities
            if entity.start is not None and entity.end is not None
        }
        counts["tp"] += len(gold_spans & predicted_spans)
        counts["predicted"] += len(predicted_spans)
        counts["gold"] += len(gold_spans)

    precision = counts["tp"] / counts["predicted"] if counts["predicted"] else 0.0
    recall = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
    micro_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )
    return NerScore(
        dataset=dataset_name,
        model=model,
        precision=precision,
        recall=recall,
        micro_f1=micro_f1,
        gold_spans=counts["gold"],
        predicted_spans=counts["predicted"],
        samples=len(texts),
        seconds=seconds,
    )


def run_eval(args: argparse.Namespace) -> list[str]:
    random.seed(SEED)
    classification_scores: list[ClassificationScore] = []
    ner_scores: list[NerScore] = []

    for model in args.models:
        print(f"[{model}] ag_news ({args.samples} samples)...", flush=True)
        classification_scores.append(
            eval_classification(
                "gliner2.5", model, "ag_news", "fancyzhx/ag_news",
                AG_NEWS_LABELS, args.samples, args.batch_size,
            )
        )
        print(f"[{model}] rotten_tomatoes ({args.samples} samples)...", flush=True)
        classification_scores.append(
            eval_classification(
                "gliner2.5", model, "rotten_tomatoes",
                "cornell-movie-review-data/rotten_tomatoes",
                ROTTEN_LABELS, args.samples, args.batch_size,
            )
        )
        print(f"[{model}] cross_ner politics ({args.samples} samples)...", flush=True)
        ner_scores.append(
            eval_ner(
                "gliner2.5", model, "cross_ner/politics", "mneb/cross-ner",
                "politics", args.samples, args.batch_size,
            )
        )

    return build_report_lines(args, classification_scores, ner_scores)


def build_report_lines(
    args: argparse.Namespace,
    classification_scores: list[ClassificationScore],
    ner_scores: list[NerScore],
) -> list[str]:
    summary_rows = []
    for model in args.models:
        by_dataset = {
            score.dataset: score
            for score in classification_scores
            if score.model == model
        }
        ner = next(score for score in ner_scores if score.model == model)
        summary_rows.append(
            f"| {model} "
            f"| {by_dataset['ag_news'].accuracy:.1%} "
            f"| {by_dataset['ag_news'].macro_f1:.3f} "
            f"| {by_dataset['rotten_tomatoes'].accuracy:.1%} "
            f"| {by_dataset['rotten_tomatoes'].macro_f1:.3f} "
            f"| {ner.micro_f1:.3f} |"
        )

    lines = [
        "# GLiNER2.5 Zero-Shot Accuracy Eval",
        "",
        f"Backend `gliner2.5`, models {', '.join(args.models)}, device `cpu`, "
        f"{args.samples} test samples per dataset (seeded shuffle, seed {SEED}), "
        "no fine-tuning and no in-context examples.",
        "",
        "## Environment",
        "",
        *environment_lines(("gliner2", "transformers", "torch", "datasets")),
        "",
        "## Summary",
        "",
        "| Model | ag_news acc | ag_news macro-F1 | rotten_tomatoes acc "
        "| rotten_tomatoes macro-F1 | CrossNER politics micro-F1 |",
        "|---|---|---|---|---|---|",
        *summary_rows,
        "",
        "## Classification",
        "",
        "Zero-shot labels: ag_news uses "
        f"{list(AG_NEWS_LABELS.values())}, rotten_tomatoes uses "
        f"{list(ROTTEN_LABELS.values())}.",
        "",
        "| Dataset | Model | Accuracy | Macro-F1 | Samples | Seconds |",
        "|---|---|---|---|---|---|",
        *[
            f"| {score.dataset} | {score.model} | {score.accuracy:.1%} "
            f"| {score.macro_f1:.3f} | {score.samples} | {score.seconds:,.1f} |"
            for score in classification_scores
        ],
        "",
        "## NER (exact span match)",
        "",
        f"CrossNER politics via `mneb/cross-ner`, threshold {NER_THRESHOLD}, using the "
        "label descriptions shipped with the dataset. A predicted span counts only "
        "when (start, end, label) all match a gold span.",
        "",
        "| Dataset | Model | Precision | Recall | Micro-F1 | Gold spans "
        "| Predicted | Samples | Seconds |",
        "|---|---|---|---|---|---|---|---|---|",
        *[
            f"| {score.dataset} | {score.model} | {score.precision:.3f} "
            f"| {score.recall:.3f} | {score.micro_f1:.3f} | {score.gold_spans} "
            f"| {score.predicted_spans} | {score.samples} | {score.seconds:,.1f} |"
            for score in ner_scores
        ],
        "",
        "## Notes",
        "",
        "- Zero-shot means the checkpoints never saw these datasets' label sets",
        "  during this eval; scores hinge on how the labels are verbalized.",
        "- Classification uses `classify_text` with a single task; the argmax",
        "  label is compared with the gold class.",
        "- NER uses exact character-offset matching, which is stricter than",
        "  token-level or partial-overlap scoring.",
        "- Sample subsets keep CPU runtime practical; expect a few points of",
        "  variance versus full test sets.",
        "- Seconds are wall-clock per dataset pass and include no model load time",
        "  for warm models; they are informational, not a benchmark.",
    ]
    return lines


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment_lines(extra_packages: tuple[str, ...] = ()) -> list[str]:
    lines = [
        f"- Date: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"- Platform: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        f"- aibackends: {_package_version('aibackends')}",
    ]
    for package in extra_packages:
        lines.append(f"- {package}: {_package_version(package)}")
    return lines


def write_report(name: str, lines: list[str]) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_prefix = datetime.now(UTC).strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"{date_prefix}_{name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["small", "base", "multi"],
        help="Model variants to evaluate (small, base, multi, or HF repo ids).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=150,
        help="Test samples per dataset.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Native batch size for inference.",
    )
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    try:
        lines = run_eval(args)
    except AIBackendsError as exc:
        raise SystemExit(f"Eval failed: {exc}") from exc

    report_path = write_report("gliner25-zero-shot", lines)
    print(f"\nReport written to {report_path}", flush=True)


if __name__ == "__main__":
    main()

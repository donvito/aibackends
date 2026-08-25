# GLiNER2.5 Zero-Shot Accuracy Eval

Backend `gliner2.5`, models small, base, multi, device `cpu`, 150 test samples per dataset (seeded shuffle, seed 20260825), no fine-tuning and no in-context examples.

## Environment

- Date: 2026-08-25 09:32:07 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Python: 3.12.3
- aibackends: 0.5.0
- gliner2: 2.0.0
- transformers: 4.57.6
- torch: 2.13.0+cpu
- datasets: 5.0.1

## Summary

| Model | ag_news acc | ag_news macro-F1 | rotten_tomatoes acc | rotten_tomatoes macro-F1 | CrossNER politics micro-F1 |
|---|---|---|---|---|---|
| small | 70.0% | 0.673 | 75.3% | 0.750 | 0.503 |
| base | 71.3% | 0.671 | 79.3% | 0.793 | 0.605 |
| multi | 73.3% | 0.700 | 69.3% | 0.691 | 0.587 |

## Classification

Zero-shot labels: ag_news uses ['world news', 'sports', 'business', 'science and technology'], rotten_tomatoes uses ['negative', 'positive'].

| Dataset | Model | Accuracy | Macro-F1 | Samples | Seconds |
|---|---|---|---|---|---|
| ag_news | small | 70.0% | 0.673 | 150 | 5.9 |
| rotten_tomatoes | small | 75.3% | 0.750 | 150 | 1.1 |
| ag_news | base | 71.3% | 0.671 | 150 | 8.0 |
| rotten_tomatoes | base | 79.3% | 0.793 | 150 | 2.9 |
| ag_news | multi | 73.3% | 0.700 | 150 | 11.2 |
| rotten_tomatoes | multi | 69.3% | 0.691 | 150 | 4.3 |

## NER (exact span match)

CrossNER politics via `mneb/cross-ner`, threshold 0.5, using the label descriptions shipped with the dataset. A predicted span counts only when (start, end, label) all match a gold span.

| Dataset | Model | Precision | Recall | Micro-F1 | Gold spans | Predicted | Samples | Seconds |
|---|---|---|---|---|---|---|---|---|
| cross_ner/politics | small | 0.458 | 0.557 | 0.503 | 948 | 1153 | 150 | 6.3 |
| cross_ner/politics | base | 0.543 | 0.681 | 0.605 | 948 | 1189 | 150 | 18.9 |
| cross_ner/politics | multi | 0.541 | 0.641 | 0.587 | 948 | 1123 | 150 | 26.5 |

## Notes

- Zero-shot means the checkpoints never saw these datasets' label sets
  during this eval; scores hinge on how the labels are verbalized.
- Classification uses `classify_text` with a single task; the argmax
  label is compared with the gold class.
- NER uses exact character-offset matching, which is stricter than
  token-level or partial-overlap scoring.
- Sample subsets keep CPU runtime practical; expect a few points of
  variance versus full test sets.
- Seconds are wall-clock per dataset pass and include no model load time
  for warm models; they are informational, not a benchmark.

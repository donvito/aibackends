# Evals

Accuracy evaluations for supported models, complementing the latency-focused
scripts in `benchmarks/`. Reports are dated markdown files in
`evals/reports/`, intended to be committed so results can be referenced on
GitHub.

## `eval_tool_calls.py`

Verifies tool-call accuracy for models with native tool calling (LFM2.5).
Each labeled case sends one completion with a tool list in the system prompt,
parses the predicted calls with `aibackends.core.tool_calls.extract_tool_calls`,
and scores:

- **Tool selection accuracy**: the set of called tool names matches the
  expected set. "No tool" cases count too — the model must answer directly
  when no tool applies.
- **Argument accuracy**: among correct selections, the arguments match after
  normalization (strings case-insensitively, numbers numerically).
- **Exact match accuracy**: both of the above.

```bash
python evals/eval_tool_calls.py --runtime llamacpp --device cpu
python evals/eval_tool_calls.py --runtime transformers --device cpu
python evals/eval_tool_calls.py --runtime llamacpp --quantization Q8_0
```

The case set covers single-tool questions (weather, currency, time), one
multi-tool question, and two no-tool questions. Latency per case is recorded
but is secondary; use `benchmarks/` for performance numbers.

Requires `aibackends[llamacpp]` or `aibackends[transformers]`. Evals run
model inference, so run them one at a time and avoid running them while a
benchmark is in flight.

## `eval_gliner25.py`

Runs a compact labeled evaluation across the GLiNER2.5 small, base, and
multilingual checkpoints through the aibackends `gliner25` backend:

- exact entity span precision, recall, and F1
- typed relation-triple precision, recall, and F1
- constrained-classification exact-match accuracy and feasibility
- span-attribute accuracy
- Joint IE graph validity and source-offset integrity

```bash
python3 evals/eval_gliner25.py --models small base multi --device cpu
```

The fixture lives in `evals/data/gliner25_cases.json`, and the dated report
contains each expected and predicted result. This is a targeted applied eval
for the repository examples, not a reproduction of Fastino's 16-dataset
research benchmark. Requires `aibackends[gliner2]`.
The latest committed comparison is
[`2026-08-25_gliner25-applied-eval-cpu.md`](reports/2026-08-25_gliner25-applied-eval-cpu.md).

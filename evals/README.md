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

Scores GLiNER 2.5 on labeled synthetic cases for the six Fastino blog use
cases: constrained agent routing and guardrails, joint knowledge-graph
triples, PII redaction, contract fields, and clinical entities/attributes.

```bash
python evals/eval_gliner25.py --device cpu --model gliner25-small
python evals/eval_gliner25.py --device cpu --model gliner25-base
```

Requires `aibackends[gliner25]`. Reports land in `evals/reports/` as
`YYYY-MM-DD_<model>-<device>.md` (for example `2026-08-25_gliner25-small-cpu.md`).
A case **passes** when every gold label is found (recall 1.0) and constraints
are feasible; extra predicted labels lower precision but do not fail the case.
Entity matches allow substring overlap after normalization.

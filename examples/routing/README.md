# Prompt routing examples

Zero-shot prompt routing with
[`LiquidAI/LFM2.5-Encoder-350M-Prompt-Router`](https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M-Prompt-Router),
a 350M bidirectional encoder fine-tuned to score a prompt against free-text
routing lanes in a single forward pass. Lanes are ordinary prose supplied at
call time, so there is nothing to train or retrain when the taxonomy changes.

The model runs through the `transformers` library with
`trust_remote_code=True` (the routing head lives in the model repo's custom
code). It is CPU-friendly: expect roughly a quarter second per prompt on a
laptop CPU after the first load. The first run downloads ~1.4 GB of weights.

## Setup

```bash
python3 -m pip install -e ".[routing]"

# Only needed for the dispatch demos:
python3 -m pip install -e ".[guardrails,pii]"   # route_and_dispatch --dispatch
python3 -m pip install -e ".[llamacpp-metal]"   # route_by_complexity local tiers
```

## Examples

One file per scenario, each self-contained. All take `--device` (default
`cpu`) and most take `--threshold`.

```bash
# The smallest possible routing example: a few lanes, ranked scores.
python3 examples/routing/route_prompt.py

# Device-assistant orchestration: separate simple function calls and tool
# use from complex multi-step agentic work.
python3 examples/routing/route_device_assistant.py

# Code-language routing: send each bug report to the right language expert.
python3 examples/routing/route_code_language.py

# Support-ticket intent classification, including multilingual tickets.
python3 examples/routing/route_support_intent.py

# Add a category on the fly: a "Soccer agent" lane claims a soccer question.
python3 examples/routing/route_custom_category.py

# Small encoder in front of capability backends: moderation and PII lanes
# dispatch to GliGuard and GLiNER when --dispatch is set.
python3 examples/routing/route_and_dispatch.py
python3 examples/routing/route_and_dispatch.py --dispatch --device cpu

# Model-tier routing by complexity: easy prompts run on LFM2.5-2.6B locally,
# moderate ones target Qwen3.8-27B, and the hardest are handed off to
# frontier cloud models (GPT-5.6 Sol, Claude Opus 5 / Fable 5, Grok 4.6).
python3 examples/routing/route_by_complexity.py
python3 examples/routing/route_by_complexity.py --skip-local
python3 examples/routing/route_by_complexity.py --run-local-tiers
```

## CLI equivalent

```bash
aibackends task route-prompt \
    --input "Can you help me debug a failing Python unit test?" \
    --labels "coding,sales,creative writing,general knowledge"
```

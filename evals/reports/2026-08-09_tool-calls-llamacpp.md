# Tool Call Accuracy Eval

Runtime `llamacpp`, model `lfm2.5-2.6b`, device `cpu`, max_tokens 1024. One completion per case; predicted calls parsed with `aibackends.core.tool_calls.extract_tool_calls`.

## Environment

- Date: 2026-08-09 07:14:45 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Python: 3.12.3
- aibackends: 0.3.0
- transformers: 5.14.1
- torch: 2.13.0+cpu
- llama-cpp-python: 0.3.34

## Metrics

| Metric | Score |
|---|---|
| Tool selection accuracy | 10/10 (100%) |
| Argument accuracy (of correct selections) | 8/8 (100%) |
| Exact match accuracy | 10/10 (100%) |
| Mean latency per case | 17,773 ms |

## Cases

| # | Question | Expected | Predicted | Selection | Arguments |
|---|---|---|---|---|---|
| 1 | What is the weather in Paris right now? | get_weather(city='Paris') | get_weather(city='Paris') | pass | pass |
| 2 | How warm is it in Tokyo today? | get_weather(city='Tokyo') | get_weather(city='Tokyo') | pass | pass |
| 3 | Is it raining in London at the moment? | get_weather(city='London') | get_weather(city='London') | pass | pass |
| 4 | Convert 100 US dollars to euros. | convert_currency(amount=100, from_currency='USD', to_currency='EUR') | convert_currency(amount=100, from_currency='USD', to_currency='EUR') | pass | pass |
| 5 | How much is 250 EUR in USD? | convert_currency(amount=250, from_currency='EUR', to_currency='USD') | convert_currency(amount=250, from_currency='EUR', to_currency='USD') | pass | pass |
| 6 | I have 75 British pounds. How many dollars is that? | convert_currency(amount=75, from_currency='GBP', to_currency='USD') | convert_currency(amount=75, from_currency='GBP', to_currency='USD') | pass | pass |
| 7 | What time is it in New York? | get_time(city='New York') | get_time(city='New York') | pass | pass |
| 8 | What's the weather in Rome, and how much is 50 USD in EUR? | get_weather(city='Rome')<br>convert_currency(amount=50, from_currency='USD', to_currency='EUR') | get_weather(city='Rome')<br>convert_currency(amount=50, from_currency='USD', to_currency='EUR') | pass | pass |
| 9 | What is the capital of France? | (no tool) | (no tool) | pass | pass |
| 10 | Write a haiku about rain. | (no tool) | (no tool) | pass | pass |

## Notes

- Tool selection counts 'no tool' cases: the model must answer
  directly when no tool applies.
- Arguments are compared after normalization (strings
  case-insensitively, numbers numerically).
- Latency includes the model's reasoning tokens, so it varies
  with question complexity.

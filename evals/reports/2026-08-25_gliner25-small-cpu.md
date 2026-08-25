# GLiNER 2.5 Use-Case Accuracy Eval

Backend `gliner25`, model `gliner25-small`, device `cpu`. Labeled synthetic cases covering the Fastino GLiNER 2.5 blog use cases. Entity and field matches allow substring overlap after whitespace/case normalization; classification requires the expected task labels.

## Environment

- Date: 2026-08-25 08:55:26 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Python: 3.12.3
- aibackends: 0.6.0
- gliner2: 2.0.0
- transformers: 4.57.6
- torch: 2.13.0
- protobuf: 7.36.0

## Metrics

| Metric | Score |
|---|---|
| Exact case pass rate | 8/8 (100%) |
| Mean precision | 0.79 |
| Mean recall | 1.00 |
| Mean F1 | 0.88 |
| Mean latency per case | 61 ms |

## Cases

| # | Use case | Case | Pass | P | R | F1 | Detail |
|---|---|---|---|---|---|---|---|
| 1 | agent-routing | route-delete-to-file-tool | pass | 1.00 | 1.00 | 1.00 | labels expected={'intent': 'delete', 'destination': 'file_tool'} predicted={'intent': 'delete', 'destination': 'file_tool'} feasible=True |
| 2 | agent-guardrails | guard-injection-is-unsafe | pass | 0.50 | 1.00 | 0.67 | labels expected={'safety': 'unsafe'} predicted={'safety': 'unsafe', 'harm_type': ['prompt_injection']} feasible=True |
| 3 | agent-guardrails | guard-birthday-is-safe | pass | 0.50 | 1.00 | 0.67 | labels expected={'safety': 'safe'} predicted={'safety': 'safe', 'harm_type': ['benign']} feasible=True |
| 4 | knowledge-graph | graph-employment-and-location | pass | 1.00 | 1.00 | 1.00 | triples predicted=['ada lovelace-works_for->fastino labs', 'charles babbage-works_for->fastino labs', 'fastino labs-located_in->london'] expected=['ada lovel... |
| 5 | pii-redaction | pii-contract-contacts | pass | 1.00 | 1.00 | 1.00 | redacted=['landlord@sampledomain.test', 'tenant@sampledomain.test'] still_visible=[] |
| 6 | extraction | ner-people-and-city | pass | 1.00 | 1.00 | 1.00 | entities predicted=['location:london', 'person:ada lovelace', 'person:charles babbage'] expected=['location:london', 'person:ada lovelace', 'person:charles b... |
| 7 | contract-review | contract-parties-and-fee | pass | 1.00 | 1.00 | 1.00 | entities predicted=['money:usd 18,500', 'organization:contoso retail inc.', 'organization:northwind analytics llc'] expected=['money:usd 18,500', 'organizati... |
| 8 | clinical-extraction | clinical-negated-fever | pass | 0.29 | 1.00 | 0.45 | entities predicted=['medication:amoxicillin 500mg capsules', 'medication:ibuprofen tablets', 'symptom:cough', 'symptom:fever', 'symptom:rash', 'symptom:sinus... |

## By use case

| Use case | Pass | Mean F1 |
|---|---|---|
| agent-routing | 1/1 | 1.00 |
| agent-guardrails | 2/2 | 0.67 |
| knowledge-graph | 1/1 | 1.00 |
| pii-redaction | 1/1 | 1.00 |
| extraction | 1/1 | 1.00 |
| contract-review | 1/1 | 1.00 |
| clinical-extraction | 1/1 | 0.45 |

## Notes

- This eval measures use-case usefulness on short synthetic texts, 
  not the 16-dataset public benchmark from the Fastino blog.
- Classification cases also require `feasible=True` when constraints 
  are declared, matching GLiNER 2.5 constrained decoding.
- Latency includes warm inference only; load the model once first.

# GLiNER2.5 Applied Use-Case Eval

Targeted source-grounded evaluation of the repository's GLiNER2.5 examples. This is not a reproduction of Fastino's 16-dataset research benchmark.

## Environment

- Date: 2026-08-25 09:15:01 UTC
- Platform: Linux-6.12.94+-x86_64-with-glibc2.39
- Processor: x86_64
- GPU: none detected
- Python: 3.12.3
- aibackends: 0.5.0
- gliner2: 2.0.0
- transformers: 4.57.6
- torch: 2.11.0
- protobuf: 7.36.0
- Device requested: cpu
- Fixture: `evals/data/gliner25_cases.json`

## Metrics

| Model | Load (s) | Entity P/R/F1 | Relation P/R/F1 | Class accuracy | Attribute accuracy | Feasible | Graph valid | Offset integrity |
|---|---:|---|---|---:|---:|---:|---:|---:|
| `small` | 4.34 | 71.4% / 76.9% / 74.1% | 100.0% / 100.0% / 100.0% | 100.0% | 50.0% | 100.0% | 100.0% | 100.0% |
| `base` | 14.69 | 71.4% / 76.9% / 74.1% | 100.0% / 100.0% / 100.0% | 100.0% | 75.0% | 100.0% | 100.0% | 100.0% |
| `multi` | 19.91 | 76.9% / 76.9% / 76.9% | 100.0% / 100.0% / 100.0% | 60.0% | 50.0% | 100.0% | 100.0% | 100.0% |

## `small` cases

Model `fastino/gliner2.5-small-v1`.

| Category | Case | Expected | Predicted | Exact | Offsets |
|---|---|---|---|---:|---:|
| entities | `product_announcement` | `[["company", "Apple", "0", "5"], ["location", "Cupertino", "46", "55"], ["person", "Tim Cook", "10", "18"], ["product", "iPhone 15", "33", "42"]]` | `[["company", "Apple", "0", "5"], ["location", "Cupertino", "46", "55"], ["person", "Tim Cook", "10", "18"], ["product", "iPhone 15", "33", "42"]]` | pass | pass |
| entities | `clinical_entities` | `[["dosage", "20 mg", "27", "32"], ["location", "Boston", "62", "68"], ["medication", "lisinopril", "33", "43"], ["person", "Amina Patel", "4", "15"]]` | `[["dosage", "20 mg", "27", "32"], ["location", "Boston", "62", "68"], ["medication", "lisinopril", "33", "43"], ["person", "Dr. Amina Patel", "0", "15"]]` | FAIL | pass |
| entities | `whole_obligation` | `[["obligation", "The tenant must return all access cards and delete every local copy of customer data within ten business days after termination.", "0", "128"]]` | `[["obligation", "delete every local copy of customer data within ten business days after termination", "44", "127"], ["obligation", "return all access cards", "16", "39"]]` | FAIL | pass |
| entities | `long_document_pii` | `[["email", "maya@example.test", "323", "340"], ["person", "Maya Chen", "295", "304"], ["phone_number", "+1 555 0100", "543", "554"], ["termination_clause", "Either party may terminate this Agreement by giving ninety days written notice.", "556", "635"]]` | `[["email", "maya@example.test", "323", "340"], ["person", "Maya Chen", "295", "304"], ["phone_number", "+1 555 0100", "543", "554"], ["termination_clause", "ninety days written notice", "608", "634"]]` | FAIL | pass |
| classification | `route_summary` | `{"route": "small_local_model", "task_type": "summarization"}` | `{"route": "small_local_model", "task_type": "summarization"}` | pass | n/a |
| classification | `route_reasoning` | `{"route": "large_reasoning_model", "task_type": "reasoning"}` | `{"route": "large_reasoning_model", "task_type": "reasoning"}` | pass | n/a |
| classification | `route_live_data` | `{"route": "tool_agent", "task_type": "live_data"}` | `{"route": "tool_agent", "task_type": "live_data"}` | pass | n/a |
| classification | `guardrail_benign` | `{"harm_type": "benign", "safety": "safe"}` | `{"harm_type": "benign", "safety": "safe"}` | pass | n/a |
| classification | `guardrail_exfiltration` | `{"harm_type": "data_exfiltration", "safety": "unsafe"}` | `{"harm_type": "data_exfiltration", "safety": "unsafe"}` | pass | n/a |
| relations | `employment_graph` | `[["located_in", "Acme Robotics", "Paris"], ["works_for", "Alice Chen", "Acme Robotics"]]` | `[["located_in", "Acme Robotics", "Paris"], ["works_for", "Alice Chen", "Acme Robotics"]]` | pass | pass |
| relations | `leadership_graph` | `[["located_in", "Apple", "Cupertino"], ["works_for", "Tim Cook", "Apple"]]` | `[["located_in", "Apple", "Cupertino"], ["works_for", "Tim Cook", "Apple"]]` | pass | pass |
| attributes | `clinical_attributes` | `[["medication", "ibuprofen", "dosage_form", "tablet"], ["symptom", "chest pain", "negation_status", "negated"], ["symptom", "nausea", "negation_status", "negated"], ["symptom", "severe headache", "negation_status", "present"]]` | `[["medication", "ibuprofen", "dosage_form", "unspecified"], ["symptom", "chest pain", "negation_status", "negated"], ["symptom", "nausea", "negation_status", "negated"], ["symptom", "severe headache", "negation_status", "negated"]]` | FAIL | pass |

## `base` cases

Model `fastino/gliner2.5-base-v1`.

| Category | Case | Expected | Predicted | Exact | Offsets |
|---|---|---|---|---:|---:|
| entities | `product_announcement` | `[["company", "Apple", "0", "5"], ["location", "Cupertino", "46", "55"], ["person", "Tim Cook", "10", "18"], ["product", "iPhone 15", "33", "42"]]` | `[["company", "Apple", "0", "5"], ["location", "Cupertino", "46", "55"], ["person", "Tim Cook", "10", "18"], ["product", "iPhone 15", "33", "42"]]` | pass | pass |
| entities | `clinical_entities` | `[["dosage", "20 mg", "27", "32"], ["location", "Boston", "62", "68"], ["medication", "lisinopril", "33", "43"], ["person", "Amina Patel", "4", "15"]]` | `[["dosage", "20 mg", "27", "32"], ["location", "Boston", "62", "68"], ["medication", "lisinopril", "33", "43"], ["person", "Dr. Amina Patel", "0", "15"]]` | FAIL | pass |
| entities | `whole_obligation` | `[["obligation", "The tenant must return all access cards and delete every local copy of customer data within ten business days after termination.", "0", "128"]]` | `[["obligation", "delete every local copy of customer data", "44", "84"], ["obligation", "return all access cards", "16", "39"]]` | FAIL | pass |
| entities | `long_document_pii` | `[["email", "maya@example.test", "323", "340"], ["person", "Maya Chen", "295", "304"], ["phone_number", "+1 555 0100", "543", "554"], ["termination_clause", "Either party may terminate this Agreement by giving ninety days written notice.", "556", "635"]]` | `[["email", "maya@example.test", "323", "340"], ["person", "Maya Chen", "295", "304"], ["phone_number", "+1 555 0100", "543", "554"], ["termination_clause", "ninety days written notice", "608", "634"]]` | FAIL | pass |
| classification | `route_summary` | `{"route": "small_local_model", "task_type": "summarization"}` | `{"route": "small_local_model", "task_type": "summarization"}` | pass | n/a |
| classification | `route_reasoning` | `{"route": "large_reasoning_model", "task_type": "reasoning"}` | `{"route": "large_reasoning_model", "task_type": "reasoning"}` | pass | n/a |
| classification | `route_live_data` | `{"route": "tool_agent", "task_type": "live_data"}` | `{"route": "tool_agent", "task_type": "live_data"}` | pass | n/a |
| classification | `guardrail_benign` | `{"harm_type": "benign", "safety": "safe"}` | `{"harm_type": "benign", "safety": "safe"}` | pass | n/a |
| classification | `guardrail_exfiltration` | `{"harm_type": "data_exfiltration", "safety": "unsafe"}` | `{"harm_type": "data_exfiltration", "safety": "unsafe"}` | pass | n/a |
| relations | `employment_graph` | `[["located_in", "Acme Robotics", "Paris"], ["works_for", "Alice Chen", "Acme Robotics"]]` | `[["located_in", "Acme Robotics", "Paris"], ["works_for", "Alice Chen", "Acme Robotics"]]` | pass | pass |
| relations | `leadership_graph` | `[["located_in", "Apple", "Cupertino"], ["works_for", "Tim Cook", "Apple"]]` | `[["located_in", "Apple", "Cupertino"], ["works_for", "Tim Cook", "Apple"]]` | pass | pass |
| attributes | `clinical_attributes` | `[["medication", "ibuprofen", "dosage_form", "tablet"], ["symptom", "chest pain", "negation_status", "negated"], ["symptom", "nausea", "negation_status", "negated"], ["symptom", "severe headache", "negation_status", "present"]]` | `[["medication", "ibuprofen", "dosage_form", "tablet"], ["symptom", "chest pain", "negation_status", "negated"], ["symptom", "headache", "negation_status", "negated"], ["symptom", "nausea", "negation_status", "negated"]]` | FAIL | pass |

## `multi` cases

Model `fastino/gliner2.5-multi-v1`.

| Category | Case | Expected | Predicted | Exact | Offsets |
|---|---|---|---|---:|---:|
| entities | `product_announcement` | `[["company", "Apple", "0", "5"], ["location", "Cupertino", "46", "55"], ["person", "Tim Cook", "10", "18"], ["product", "iPhone 15", "33", "42"]]` | `[["company", "Apple", "0", "5"], ["location", "Cupertino", "46", "55"], ["person", "Tim Cook", "10", "18"], ["product", "iPhone 15", "33", "42"]]` | pass | pass |
| entities | `clinical_entities` | `[["dosage", "20 mg", "27", "32"], ["location", "Boston", "62", "68"], ["medication", "lisinopril", "33", "43"], ["person", "Amina Patel", "4", "15"]]` | `[["dosage", "20 mg", "27", "32"], ["location", "Boston", "62", "68"], ["medication", "lisinopril", "33", "43"], ["person", "Dr. Amina Patel", "0", "15"]]` | FAIL | pass |
| entities | `whole_obligation` | `[["obligation", "The tenant must return all access cards and delete every local copy of customer data within ten business days after termination.", "0", "128"]]` | `[["obligation", "access cards", "27", "39"]]` | FAIL | pass |
| entities | `long_document_pii` | `[["email", "maya@example.test", "323", "340"], ["person", "Maya Chen", "295", "304"], ["phone_number", "+1 555 0100", "543", "554"], ["termination_clause", "Either party may terminate this Agreement by giving ninety days written notice.", "556", "635"]]` | `[["email", "maya@example.test", "323", "340"], ["person", "Maya Chen", "295", "304"], ["phone_number", "+1 555 0100", "543", "554"], ["termination_clause", "ninety days written notice", "608", "634"]]` | FAIL | pass |
| classification | `route_summary` | `{"route": "small_local_model", "task_type": "summarization"}` | `{"route": "small_local_model", "task_type": "summarization"}` | pass | n/a |
| classification | `route_reasoning` | `{"route": "large_reasoning_model", "task_type": "reasoning"}` | `{"route": "large_reasoning_model", "task_type": "reasoning"}` | pass | n/a |
| classification | `route_live_data` | `{"route": "tool_agent", "task_type": "live_data"}` | `{"route": "small_local_model", "task_type": "summarization"}` | FAIL | n/a |
| classification | `guardrail_benign` | `{"harm_type": "benign", "safety": "safe"}` | `{"harm_type": "benign", "safety": "safe"}` | pass | n/a |
| classification | `guardrail_exfiltration` | `{"harm_type": "data_exfiltration", "safety": "unsafe"}` | `{"harm_type": "prompt_injection", "safety": "unsafe"}` | FAIL | n/a |
| relations | `employment_graph` | `[["located_in", "Acme Robotics", "Paris"], ["works_for", "Alice Chen", "Acme Robotics"]]` | `[["located_in", "Acme Robotics", "Paris"], ["works_for", "Alice Chen", "Acme Robotics"]]` | pass | pass |
| relations | `leadership_graph` | `[["located_in", "Apple", "Cupertino"], ["works_for", "Tim Cook", "Apple"]]` | `[["located_in", "Apple", "Cupertino"], ["works_for", "Tim Cook", "Apple"]]` | pass | pass |
| attributes | `clinical_attributes` | `[["medication", "ibuprofen", "dosage_form", "tablet"], ["symptom", "chest pain", "negation_status", "negated"], ["symptom", "nausea", "negation_status", "negated"], ["symptom", "severe headache", "negation_status", "present"]]` | `[["medication", "ibuprofen", "dosage_form", "capsule"], ["symptom", "chest pain", "negation_status", "negated"], ["symptom", "headache", "negation_status", "present"], ["symptom", "nausea", "negation_status", "negated"]]` | FAIL | pass |

## Notes

- Entity and relation metrics use exact label, source text, and character spans.
- Classification accuracy requires the complete constrained assignment to match.
- Attribute accuracy counts exact expected entity, attribute, and value matches.
- Graph validity checks feasibility, typed endpoints, no self-loops, and unique heads.
- Offset integrity requires every returned span to slice back to identical source text.
- Quality numbers apply only to this compact fixture; tune schemas on domain data.

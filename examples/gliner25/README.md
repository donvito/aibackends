# GLiNER2.5 examples

Runnable demos of the [GLiNER2.5](https://fastino.ai/blog/gliner2-5-span-free-information-extraction)
capabilities through the `aibackends` extraction tasks (`extract_entities`,
`classify_text`, `extract_graph`).

## Setup

```bash
pip install 'aibackends[extraction]'
```

Model variants: `small` (74M, fastest on CPU), `base` (194M, default English),
`multi` (287M, multilingual). Every script accepts `--model` and `--device`.
Models are downloaded from Hugging Face on first use and cached per process.

## Scripts

```bash
python3 examples/gliner25/entity_extraction.py            # PII detection + offset redaction
python3 examples/gliner25/unlimited_spans.py              # clause-length entities
python3 examples/gliner25/knowledge_graph.py              # joint entity-relation graph
python3 examples/gliner25/constrained_classification.py   # agent routing with constraints
python3 examples/gliner25/span_attributes.py              # clinical extraction, qualified spans
python3 examples/gliner25/long_document.py                # full-contract chunked extraction
python3 examples/gliner25/multilingual_ner.py             # zero-shot multilingual NER
```

`long_document.py` reads `examples/data/sample_contract.txt`. The CLI mirrors
the same tasks:

```bash
aibackends task extract-entities --input examples/data/sample_contract.txt \
    --labels party,monetary_amount --model small
aibackends task classify-text --input "Refund my card" --labels billing,bug,feature
aibackends task extract-graph --input "Alice works for Acme in Paris." \
    --entities person,organization,location \
    --relation works_for:person:organization --relation located_in:organization:location
```

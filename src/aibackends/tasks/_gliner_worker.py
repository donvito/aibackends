from __future__ import annotations

import json
import sys

from gliner import GLiNER


def main() -> int:
    payload = json.load(sys.stdin)
    model = GLiNER.from_pretrained(payload["model_id"])
    entities = model.predict_entities(
        payload["text"],
        payload["labels"],
        threshold=payload["threshold"],
    )
    json.dump(entities, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

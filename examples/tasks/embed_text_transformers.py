import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import MINILM_L6
from aibackends.runtimes import TRANSFORMERS
from aibackends.tasks import EmbedTask, create_task


def main() -> None:
    try:
        task = create_task(
            EmbedTask,
            runtime=TRANSFORMERS,
            model=MINILM_L6,
        )
        text = (
            "After our midnight game launch, thousands of players rushed to buy the limited "
            "Lunar Dragon skin. Some purchases were charged twice, the item never appeared "
            "in the players' inventories, and the support chatbot kept suggesting cache "
            "resets instead of refunds. The operations team needs embeddings for reports "
            "like this so they can group similar incidents and spot the biggest issues fast."
        )
        vector = task.run(text)

        print("MiniLM embeddings with Transformers")
        print(f"Text: {text}")
        print(f"Embedding dimension: {len(vector)}")
        preview = ", ".join(f"{value:.4f}" for value in vector[:5])
        print(f"Embedding preview: [{preview}, ...]")
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

import sys

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import MINILM_L6
from aibackends.runtimes import TRANSFORMERS
from aibackends.workflows import EmbeddingSimilarityWorkflow, create_workflow


def main() -> None:
    try:
        workflow = create_workflow(
            EmbeddingSimilarityWorkflow,
            runtime=TRANSFORMERS,
            model=MINILM_L6,
        )
        texts = [
            (
                "Customer says their Pro subscription was charged twice after renewal and "
                "they want the extra payment refunded."
            ),
            (
                "User reports duplicate billing after upgrading to the annual plan and asks "
                "whether the second charge can be reversed."
            ),
            (
                "Customer cannot sign in because the one-time verification code never "
                "arrives, and password reset keeps failing."
            ),
            (
                "User says account access is blocked after several reset attempts, and they "
                "are not receiving the MFA email needed to log in."
            ),
        ]
        result = workflow.run(texts)

        print("MiniLM embedding similarity workflow with Transformers")
        print("Practical use case: group similar support tickets for triage")
        print(f"Texts compared: {len(result.texts)}")
        print(f"Embedding dimension: {result.embeddings[0].dimension}")
        print()
        for item in result.embeddings:
            print(f"[{item.index}] {item.text}")
        print()
        print("Ranked cosine similarity pairs:")
        for pair in result.ranked_pairs:
            print(
                f"- [{pair.left_index}] vs [{pair.right_index}] "
                f"=> {pair.cosine_similarity:.4f}"
            )
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

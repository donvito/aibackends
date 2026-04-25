import sys
from pathlib import Path

from pydantic import BaseModel

from aibackends.core.exceptions import AIBackendsError
from aibackends.models import GEMMA4_E2B
from aibackends.runtimes import LLAMACPP
from aibackends.steps.enrich import LLMAnalyser
from aibackends.steps.ingest import FileIngestor
from aibackends.steps.validate import PydanticValidator
from aibackends.workflows import Pipeline


class LeadBrief(BaseModel):
    name: str
    company: str | None = None
    email: str | None = None
    priority: str | None = None
    next_step: str | None = None


class LeadIntakePipeline(Pipeline):
    steps = [
        FileIngestor(),
        LLMAnalyser(schema=LeadBrief, prompt="Extract the lead details from the note."),
        PydanticValidator(schema=LeadBrief),
    ]


def main() -> None:
    try:
        lead_note_path = Path(__file__).parent.parent / "data" / "lead_note.txt"
        workflow = LeadIntakePipeline(runtime=LLAMACPP, model=GEMMA4_E2B)
        result = workflow.run(lead_note_path)
        print(result.model_dump_json(indent=2))
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

"""Resume redact + summarize + role-match pipeline.

Outputs ATS-friendly JSON by composing built-in workflow steps.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from aibackends.schemas.pii import Classification
from aibackends.steps.enrich import (
    LLMTextGenerator,
    PIIRedactor as PIIRedactorStep,
    TaskRunner,
)
from aibackends.steps.ingest import FileIngestor
from aibackends.workflows import Pipeline

class PIISummary(BaseModel):
    backend: str
    entities_redacted: int
    redaction_map: dict[str, str]


class ResumeMatchResult(BaseModel):
    source: str
    redacted_text: str
    summary: str
    role_match: Classification
    pii: PIISummary

RESUME_PII_LABELS = [
    "person_name",
    "email",
    "phone_number",
    "address",
    "date_of_birth",
    "url",
    "linkedin_profile",
    "github_profile",
    "company_name",
    "job_title",
    "school",
    "identification_number",
]

RESUME_SYSTEM_PROMPT = (
    "You are an expert resume reviewer helping a hiring manager triage candidates."
)

RESUME_USER_PROMPT = (
    "Summarize the redacted resume below for a hiring manager. Focus on:\n"
    "- Years and breadth of experience\n"
    "- Core technical and domain skills\n"
    "- Notable achievements or projects\n"
    "- Education highlights\n"
    "- Overall seniority and likely role fit\n\n"
    "Keep it under 200 words. Do not invent details for fields that have been "
    "redacted (they will appear as bracketed placeholders such as [person_name_1])."
)

JOB_OPENINGS: dict[str, str] = {
    "senior-backend-engineer": (
        "5+ years building distributed backend systems in Python or Go. "
        "Strong API design, databases, and production reliability skills."
    ),
    "ml-engineer": (
        "Hands-on ML/AI background: model training, evaluation, and shipping "
        "ML features to production. Comfortable with PyTorch, transformers, and MLOps."
    ),
    "platform-engineer": (
        "Kubernetes, CI/CD, observability, and developer-platform tooling. "
        "Has built internal platforms that other engineers depend on."
    ),
    "data-engineer": (
        "ETL/ELT pipelines, data warehousing, and streaming. Strong SQL plus "
        "experience with tools like Airflow, dbt, Spark, or Kafka."
    ),
    "engineering-manager": (
        "5+ years engineering experience plus 2+ years people management. "
        "Leads multi-engineer teams and balances delivery with growth."
    ),
}

ROLE_MATCH_SYSTEM_PROMPT = (
    "You are a technical recruiter matching candidates to open roles."
)

ROLE_MATCH_PROMPT = (
    "Pick the single best-fitting role for the candidate based on the redacted "
    "resume below. Base your judgement only on evidence in the resume; do not "
    "infer from redacted placeholders."
)


class ResumeRoleMatcher(Pipeline):
    steps = [
        FileIngestor(),
        PIIRedactorStep(backend="gliner", labels=RESUME_PII_LABELS),
        LLMTextGenerator(
            task_name="resume-summarize",
            system_prompt=RESUME_SYSTEM_PROMPT,
            prompt=RESUME_USER_PROMPT,
            input_key="text",
            output_key="summary",
        ),
        TaskRunner(
            task_name="classify",
            input_key="text",
            output_key="role_match",
            task_config={
                "labels": list(JOB_OPENINGS),
                "label_descriptions": JOB_OPENINGS,
                "system_prompt": ROLE_MATCH_SYSTEM_PROMPT,
                "prompt": ROLE_MATCH_PROMPT,
            },
        ),
    ]


if __name__ == "__main__":
    resume_path = Path(__file__).parent.parent / "data" / "pdf" / "resume-sample.pdf"

    workflow = ResumeRoleMatcher(runtime="llamacpp", model="gemma4-e2b")
    raw_result = workflow.run(resume_path)
    redaction = raw_result["pii_redaction"]
    result = ResumeMatchResult(
        source=str(raw_result.get("path", "")),
        redacted_text=raw_result.get("text", ""),
        summary=raw_result["summary"],
        role_match=raw_result["role_match"],
        pii=PIISummary(
            backend=redaction.backend_used,
            entities_redacted=len(redaction.entities_found),
            redaction_map=redaction.redaction_map,
        ),
    )

    print(result.model_dump_json(indent=2))

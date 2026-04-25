"""Resume redact + summarize pipeline.

Framework-first Python equivalent of this two-step CLI recipe:

    aibackends task redact-pii --input resume.pdf --backend gliner \\
      | jq -r '.redacted_text' > resume_redacted.txt
    aibackends task summarize --input "$(cat resume_redacted.txt)" \\
      --runtime llamacpp --model gemma4-e2b

This example keeps the custom logic small by composing built-in workflow steps:
file ingest, PII redaction, and prompt-driven text generation.

Requires:
    pip install 'aibackends[pdf]'         # PyMuPDF for PDF text extraction
    pip install 'aibackends[pii]'         # GLiNER PII backend
    pip install 'aibackends[llamacpp]'    # local LLM runtime used below
"""

from __future__ import annotations

from pathlib import Path

from aibackends.steps.enrich import LLMTextGenerator, PIIRedactor as PIIRedactorStep
from aibackends.steps.ingest import FileIngestor
from aibackends.workflows import Pipeline

# Custom GLiNER entity types tuned for a resume. GLiNER is open-vocabulary,
# so any descriptive label works; these names are what GLiNER will tag and
# what shows up in the redaction map (e.g. "[person_name_1]").
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

# Custom prompt for resume triage rather than a generic prose summary.
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


class ResumeRedactSummarizer(Pipeline):
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
    ]


if __name__ == "__main__":
    resume_path = Path(__file__).parent.parent / "data" / "pdf" / "resume-sample.pdf"

    workflow = ResumeRedactSummarizer(runtime="llamacpp", model="gemma4-e2b")
    result = workflow.run(resume_path)

    redaction = result["pii_redaction"]
    print(f"Redacted {len(redaction.entities_found)} PII entities "
          f"using backend '{redaction.backend_used}'.\n", flush=True)

    print("Redacted text (first 300 chars):", flush=True)
    print(result["text"][:300] + "...\n", flush=True)

    print("Summary:", flush=True)
    print(result["summary"], flush=True)

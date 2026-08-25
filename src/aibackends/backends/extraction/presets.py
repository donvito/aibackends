from __future__ import annotations

from aibackends.schemas.extraction import (
    AttributeSpec,
    ClassificationConstraint,
    ClassificationTaskSpec,
    ConstrainedClassificationSchema,
    RelationSpec,
)

ROUTING_SCHEMA = ConstrainedClassificationSchema(
    tasks=[
        ClassificationTaskSpec(
            name="intent",
            labels=["summarize", "extract", "code", "search", "chat"],
        ),
        ClassificationTaskSpec(
            name="destination",
            labels=["small_model", "large_model", "code_agent", "search_agent"],
        ),
    ],
    constraints=[
        ClassificationConstraint(
            kind="implies",
            source=("intent", "code"),
            target=("destination", "code_agent"),
        ),
        ClassificationConstraint(
            kind="implies",
            source=("intent", "search"),
            target=("destination", "search_agent"),
        ),
        ClassificationConstraint(
            kind="implies",
            source=("intent", "summarize"),
            target=("destination", "small_model"),
        ),
        ClassificationConstraint(
            kind="excludes",
            source=("intent", "chat"),
            target=("destination", "code_agent"),
        ),
        ClassificationConstraint(
            kind="excludes",
            source=("intent", "extract"),
            target=("destination", "code_agent"),
        ),
    ],
)

GUARDRAIL_SCHEMA = ConstrainedClassificationSchema(
    tasks=[
        ClassificationTaskSpec(name="safety", labels=["allow", "block"]),
        ClassificationTaskSpec(
            name="harm_type",
            labels=["prompt_injection", "pii_leak", "policy_violation", "none"],
        ),
    ],
    constraints=[
        ClassificationConstraint(
            kind="implies",
            source=("safety", "allow"),
            target=("harm_type", "none"),
        ),
        ClassificationConstraint(
            kind="excludes",
            source=("safety", "allow"),
            target=("harm_type", "prompt_injection"),
        ),
        ClassificationConstraint(
            kind="excludes",
            source=("safety", "allow"),
            target=("harm_type", "pii_leak"),
        ),
        ClassificationConstraint(
            kind="excludes",
            source=("safety", "allow"),
            target=("harm_type", "policy_violation"),
        ),
        ClassificationConstraint(
            kind="excludes",
            source=("safety", "block"),
            target=("harm_type", "none"),
        ),
    ],
)

MEMORY_GRAPH_ENTITIES = ["person", "organization", "project", "commitment", "location"]
MEMORY_GRAPH_RELATIONS = [
    RelationSpec(
        name="works_for",
        head_type="person",
        tail_type="organization",
        unique_head=True,
    ),
    RelationSpec(name="works_on", head_type="person", tail_type="project"),
    RelationSpec(
        name="owns",
        head_type="organization",
        tail_type="project",
        unique_head=True,
    ),
    RelationSpec(name="committed_to", head_type="person", tail_type="commitment"),
    RelationSpec(
        name="located_in",
        head_type="organization",
        tail_type="location",
        unique_head=True,
    ),
]

CONTRACT_LABELS = {
    "party": "Named contracting parties such as landlord, tenant, buyer, or seller",
    "obligation": "Duties a party must perform, including payment, notice, and repair",
    "termination_clause": "Language describing how or when the agreement may end",
    "date": "Calendar dates or durations such as 30 days",
    "amount": "Money amounts including currency",
    "address": "Full postal addresses of parties or the property",
}

CLINICAL_LABELS = {
    "symptom": "Reported symptoms or complaints",
    "condition": "Diagnoses or named medical conditions",
    "medication": "Drug or pharmaceutical names",
    "dosage": "Amounts such as 400mg, 2 tablets, or 5ml",
}
CLINICAL_ATTRIBUTES = [
    AttributeSpec(
        name="negation",
        labels=["affirmed", "negated"],
        applies_to=["symptom", "condition"],
        qualify_labels=True,
    ),
    AttributeSpec(
        name="dosage_form",
        labels=["tablet", "capsule", "liquid", "injection", "topical"],
        applies_to=["medication"],
        qualify_labels=True,
    ),
]

PII_LABELS = (
    "person",
    "email",
    "phone number",
    "address",
    "social security number",
    "credit card number",
    "date of birth",
    "organization",
)

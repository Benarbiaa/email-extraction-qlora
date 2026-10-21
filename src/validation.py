"""Schema + labeling-rule validation, shared by data generation and evaluation."""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "data" / "schema.json"

FIELDS = [
    "email_type", "company", "job_title", "contract_type", "work_mode",
    "stage", "sender_role", "action_required", "deadline", "next_step",
]

# labeling_rules.md §2.1
ALLOWED_STAGES = {
    "application_ack": {"applied"},
    "interview_invite": {"screening", "technical_interview", "final_interview"},
    "assessment_request": {"screening", "technical_interview"},
    "scheduling": {"screening", "technical_interview", "final_interview"},
    "info_request": {"applied", "screening", "technical_interview", "final_interview", "offer"},
    "rejection": {"closed"},
    "offer": {"offer"},
    "networking_cold_outreach": {None},
}


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def schema_errors(record: object) -> list[str]:
    """JSON Schema violations (types, enums, required fields, deadline/action rule)."""
    errors = []
    for e in _validator().iter_errors(record):
        path = ".".join(str(p) for p in e.absolute_path) or "<root>"
        errors.append(f"schema:{path}: {e.message}")
    if not errors and record.get("deadline"):
        try:
            date.fromisoformat(record["deadline"])
        except ValueError:
            errors.append(f"schema:deadline: not a real calendar date: {record['deadline']}")
    return errors


def rule_errors(record: dict, sent: date | None = None) -> list[str]:
    """Cross-field labeling-rule violations. Assumes the record is schema-valid."""
    errors = []
    allowed = ALLOWED_STAGES.get(record["email_type"])
    if allowed is not None and record["stage"] not in allowed:
        errors.append(f"rule:stage {record['stage']!r} not allowed for {record['email_type']}")
    if record["email_type"] == "rejection" and record["action_required"]:
        errors.append("rule:rejection with action_required=true")
    if sent and record["deadline"]:
        delta = (date.fromisoformat(record["deadline"]) - sent).days
        if delta < 0:
            errors.append(f"rule:deadline {record['deadline']} is before send date {sent}")
        elif delta > 90:
            errors.append(f"rule:deadline {record['deadline']} is >90 days after send date {sent}")
    return errors


def validate(record: object, sent: date | None = None) -> list[str]:
    errors = schema_errors(record)
    if errors:
        return errors
    return rule_errors(record, sent)


def canonical(record: dict) -> dict:
    """Return the record with keys in canonical order (the training target format)."""
    return {k: record[k] for k in FIELDS}

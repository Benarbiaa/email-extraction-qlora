"""Prompt construction shared by training, baselines and inference.

Two system prompts:
- SYSTEM_PROMPT: a condensed version of data/labeling_rules.md (~520 tokens). Used by the
  base-model baselines (zero-shot and few-shot), which need the rules spelled out.
- FINETUNE_SYSTEM_PROMPT: one line. Used to train and run the fine-tuned model, which learns
  the rules from the data. With the full prompt, ~20% of training examples would exceed the
  1024-token budget and get their target truncated.
Each model is thus evaluated in its intended configuration (disclosed in the README).
"""

from __future__ import annotations

import json

from src.validation import FIELDS

SYSTEM_PROMPT = """You extract a JSON record from a recruitment/HR email (French, English or mixed) received by a job candidate. Output ONLY one JSON object with exactly these keys, in this order:

email_type: application_ack | interview_invite | assessment_request | scheduling | info_request | rejection | offer | networking_cold_outreach | other
  (precedence if several apply: offer > rejection > assessment_request > interview_invite > scheduling > info_request > networking_cold_outreach > application_ack > other; scheduling = logistics of an already agreed round; cold outreach = candidate did not apply)
company: hiring company core brand name (no legal suffix, no France/Group/Solutions/Conseil/& Associés...) or null; never the ATS or job board
job_title: role as written, without Stage/PFE/Alternance/CDI/(H/F)/codes/city, or null
contract_type: internship | apprenticeship | CDI | CDD | freelance | null (only if stated)
work_mode: onsite | hybrid | remote | null (the job's arrangement, NOT the interview format)
stage: applied | screening | technical_interview | final_interview | offer | closed | null
  (first call of a process = screening; final only if explicitly last round; rejection = closed; no candidacy = null)
sender_role: recruiter | hiring_manager | HR | automated_system | other (a signature title decides)
action_required: true if the email asks the candidate to do something, else false
deadline: YYYY-MM-DD due date of the requested action (not the meeting date), resolved from the Date header, else null; must be null if action_required is false
next_step: short English phrase (max 120 chars) describing the candidate's next step, or null

Use only what the email says; unsupported fields are null."""


FINETUNE_SYSTEM_PROMPT = "Extract the recruitment email record as JSON (schema v1)."


def user_message(email: str) -> str:
    return f"Email:\n{email}"


def target_json(label: dict) -> str:
    """Canonical training target: keys in schema order, compact, UTF-8."""
    return json.dumps({k: label[k] for k in FIELDS}, ensure_ascii=False)


def chat_messages(email: str, few_shot: list[dict] | None = None, system: str = SYSTEM_PROMPT) -> list[dict]:
    """Messages for inference. few_shot: records with 'email' and 'label' (from train only)."""
    msgs = [{"role": "system", "content": system}]
    for ex in few_shot or []:
        msgs.append({"role": "user", "content": user_message(ex["email"])})
        msgs.append({"role": "assistant", "content": target_json(ex["label"])})
    msgs.append({"role": "user", "content": user_message(email)})
    return msgs


def training_example(record: dict) -> dict:
    """One SFT example in TRL's conversational prompt-completion format (loss on the completion only)."""
    return {"prompt": chat_messages(record["email"], system=FINETUNE_SYSTEM_PROMPT),
            "completion": [{"role": "assistant", "content": target_json(record["label"])}]}


# Fixed few-shot set for the few-shot baseline: one short train email per type, deterministic.
FEW_SHOT_TYPES = ["interview_invite", "rejection", "assessment_request"]


def select_few_shot(train: list[dict], types: list[str] = FEW_SHOT_TYPES, max_words: int = 90) -> list[dict]:
    picked = []
    for t in types:
        cands = sorted((r for r in train if r["label"]["email_type"] == t and len(r["email"].split()) <= max_words),
                       key=lambda r: r["id"])
        if cands:
            picked.append(cands[0])
    return picked

"""Generate synthetic recruitment emails + gold labels in batches.

Python samples a *spec* per example (email type, language, sender, fictional company,
edge cases, which fields to mention/omit); an LLM sub-agent writes the email and its label.
Each record is validated against data/schema.json, the labeling rules, and its spec.

Two steps (no API calls; see scripts/_common.py):
    emit    sample specs -> <split>/work/pending/bNNN.json + the rendered prompt
    ingest  validate <split>/work/responses/bNNN.json -> examples.jsonl / rejected.jsonl

Splits (each has its own output folder under data/generated/):
    train        generator profile A, train name pools
    test_indist  generator profile A, test-only name pools  (in-distribution test slice)
    test_xgen    generator profile B, test-only name pools  (cross-generator test slice)

Usage:
    python scripts/generate.py emit --split train --n 100 --seed 1
    python scripts/generate.py ingest --split train
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import shutil
import time
import unicodedata
from collections import Counter
from datetime import datetime, timedelta
from email.utils import format_datetime, parsedate_to_datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import GEN_DIR, append_jsonl, est_tokens, log_usage, parse_json, write_json  # noqa: E402
from src.matching import canonical_company, normalize_company, normalize_text  # noqa: E402
from src.validation import canonical, validate  # noqa: E402

SPLIT_PROFILE = {"train": "A", "test_indist": "A", "test_xgen": "B"}
PROMPT_PATH = ROOT / "data" / "prompts" / "generation_system.md"
RULES_PATH = ROOT / "data" / "labeling_rules.md"

# --------------------------------------------------------------------------- spec sampling

TYPE_WEIGHTS = {
    "application_ack": 14, "interview_invite": 14, "assessment_request": 10,
    "scheduling": 12, "info_request": 9, "rejection": 16, "offer": 6,
    "networking_cold_outreach": 11, "other": 8,
}

SENDER_ROLE_WEIGHTS = {
    "application_ack": {"automated_system": 70, "recruiter": 20, "HR": 10},
    "interview_invite": {"recruiter": 55, "hiring_manager": 30, "HR": 10, "automated_system": 5},
    "assessment_request": {"automated_system": 50, "recruiter": 40, "hiring_manager": 10},
    "scheduling": {"recruiter": 50, "hiring_manager": 35, "HR": 5, "automated_system": 10},
    "info_request": {"HR": 45, "recruiter": 45, "hiring_manager": 10},
    "rejection": {"automated_system": 45, "recruiter": 40, "hiring_manager": 10, "HR": 5},
    "offer": {"HR": 35, "recruiter": 35, "hiring_manager": 30},
    "networking_cold_outreach": {"recruiter": 70, "hiring_manager": 25, "other": 5},
    "other": {"automated_system": 60, "other": 25, "HR": 15},
}

# Target stage per type (the label must still follow what the email says).
STAGE_WEIGHTS = {
    "application_ack": {"applied": 1},
    "interview_invite": {"screening": 45, "technical_interview": 40, "final_interview": 15},
    "assessment_request": {"screening": 65, "technical_interview": 35},
    "scheduling": {"screening": 40, "technical_interview": 45, "final_interview": 15},
    "info_request": {"applied": 20, "screening": 30, "technical_interview": 20, "final_interview": 10, "offer": 20},
    "rejection": {"closed": 1},
    "offer": {"offer": 1},
    "networking_cold_outreach": {None: 1},
}

OTHER_SUBTYPES = {  # subtype -> target stage
    "job_alert_digest": None, "career_event_invite": None, "job_board_newsletter": None,
    "candidate_withdrawal_confirmation": "closed", "onboarding_logistics": "offer",
}

# Edge case -> email types it may apply to (None = any).
EDGE_CASES = {
    "ats_template": None,  # forced iff sender is automated_system
    "vague_rejection": {"rejection"},
    "relative_deadline": {"assessment_request", "info_request", "interview_invite", "scheduling", "offer"},
    "timezone": {"interview_invite", "scheduling", "assessment_request", "networking_cold_outreach"},
    "forwarded_two_companies": None,
    "reply_thread": {"scheduling", "info_request", "offer", "rejection", "interview_invite"},
    "missing_fields": None,
    "fr_en_mix": None,
    "typos": None,
    "informal_startup": {"interview_invite", "scheduling", "networking_cold_outreach", "offer", "rejection", "info_request"},
    "agency_confidential_client": {"networking_cold_outreach", "interview_invite", "scheduling", "rejection"},
    "interview_format_trap": {"interview_invite", "scheduling"},
    "multi_purpose": {"assessment_request", "interview_invite", "info_request", "offer"},
    "signature_noise": None,
    "role_switch": {"interview_invite"},
}

# Sampling weight among applicable edge cases (the ones from the project brief are boosted).
EDGE_WEIGHTS = {"relative_deadline": 5, "timezone": 4, "vague_rejection": 4, "forwarded_two_companies": 3,
                "missing_fields": 3, "fr_en_mix": 3, "typos": 3, "informal_startup": 3,
                "interview_format_trap": 3, "reply_thread": 2, "agency_confidential_client": 2,
                "multi_purpose": 2, "signature_noise": 1, "role_switch": 1}

CONTRACT_WEIGHTS = {"internship": 50, "apprenticeship": 15, "CDI": 25, "CDD": 5, "freelance": 5}
WORK_MODE_WEIGHTS = {"onsite": 30, "hybrid": 50, "remote": 20}
MENTION_PROB = {"company": 0.9, "job_title": 0.8, "contract_type": 0.55, "work_mode": 0.3}
DEADLINE_PROB = {"assessment_request": 0.7, "info_request": 0.5, "interview_invite": 0.35,
                 "scheduling": 0.3, "offer": 0.5}

LANG_WEIGHTS = {"fr": 55, "en": 25, "mixed_fr_en": 20}
REGISTERS = ["corporate_formal", "neutral", "warm", "terse"]
# Body length targets in words. Templates are never short.
LENGTHS = {"short": (5, 40), "medium": (40, 140), "long": (140, 350)}
LENGTH_WEIGHTS = {"human": {"short": 38, "medium": 40, "long": 22}, "automated": {"medium": 60, "long": 40}}

ROLE_DOMAINS = [
    "data engineering", "data analysis / BI", "data science", "machine learning engineering",
    "MLOps", "backend development", "full-stack development", "DevOps / cloud", "software engineering",
    "analytics engineering", "embedded software", "cybersecurity", "QA / test automation",
    "product management (tech)", "digital marketing",  # a few non-core roles for diversity
]
SECTORS = ["fintech", "retail", "energy", "healthtech", "logistics", "insurance", "aerospace",
           "SaaS B2B", "e-commerce", "consulting / ESN", "mobility", "media", "agritech",
           "banking", "public sector", "gaming", "telecom", "cybersecurity", "edtech"]

# Fictional names: syllable-built companies, mixed-origin person names.
CO_HEAD = ["Nova", "Kel", "Orbi", "Lumen", "Vire", "Talo", "Brin", "Quar", "Zeph", "Ardo", "Mira",
           "Sola", "Hexa", "Corva", "Ilia", "Peri", "Vela", "Opta", "Dune", "Riva", "Strato", "Alto",
           "Ondi", "Kora", "Fen", "Lyra", "Maré", "Tessel", "Arbo", "Cirra"]
CO_TAIL = ["tek", "vio", "flow", "cloud", "line", "data", "nexa", "ware", "logic", "sys", "go",
           "lab", "ia", "pulse", "grid", "metrics", "mind", "bridge", "scale", "path"]
CO_SUFFIX = ["", "", "", " Solutions", " Group", " Technologies", " Analytics", " Consulting",
             " Digital", " Systems", " & Co", " France", " Labs"]
TLDS = [".fr", ".com", ".io", ".ai", ".eu", ".tech", ".co"]
FIRST = ["Léa", "Inès", "Camille", "Manon", "Chloé", "Sarah", "Yasmine", "Emma", "Nour", "Clara",
         "Julie", "Amina", "Priya", "Sofia", "Hana", "Thomas", "Lucas", "Hugo", "Yanis", "Karim",
         "Mehdi", "Antoine", "Julien", "Maxime", "Rayan", "Omar", "Nicolas", "Tom", "Alexandre",
         "Wei", "Diego", "Arjun", "Mamadou", "Elif", "Lina", "Bastien", "Céline", "Sami", "Nadia"]
LAST = ["Martin", "Garnier", "Benali", "Lefèvre", "Moreau", "Haddad", "Rousseau", "Nguyen", "Diallo",
        "Fontaine", "Mercier", "Chevalier", "Bouzid", "Lambert", "Girard", "Perrin", "Nair", "Okafor",
        "Kaya", "Rossi", "Dubois", "Meyer", "Benoît", "Traoré", "Chen", "Laurent", "Faure", "Belkacem"]
CANDIDATES = ["Yanis", "Sarah", "Amine", "Léa", "Mehdi", "Camille", "Nour", "Hugo", "Inès", "Rayan",
              "Chloé", "Karim", "Emma", "Ilyes", "Manon"]

# Train-only name diversity (labeling rules §3): multi-word French-style names, acronyms, and
# sender domains that differ from the name, so "company = the word in the domain" isn't a shortcut.
# Patterns avoid the descriptor words that §3 strips (Group, Solutions, ...), so gold names are unambiguous.
FR_NAME_WORDS = ["Valmont", "Delorme", "Brisac", "Ferrand", "Montclair", "Aubrac", "Lavergne", "Clairval",
                 "Hautefort", "Marsan", "Rochebrune", "Vauclair", "Bellerive", "Castagne", "Duvernet",
                 "Malbosc", "Pradelle", "Sorbier", "Tavernier", "Verneuil"]
FR_PATTERNS = ["{w} & Associés", "Atelier {w}", "{w} Conseil", "Maison {w}", "{w} Industries",
               "Mutuelle {w}", "{w} Énergies", "Laboratoires {w}", "Cabinet {w}", "{w} Logistique",
               "Banque {w}", "{w} Assurances"]
RELAY_DOMAINS = ["mail.hireboard-ats.com", "jobs.talentrelay-mail.net", "notifications.recrutonline-ats.fr",
                 "candidats.jobpilot-mail.eu"]

# Test-only pools: disjoint from the train pools above, so no test company/person name
# ever appears in train (prevents name -> label memorization).
TEST_CO_HEAD = ["Zir", "Obel", "Quen", "Ysal", "Brava", "Cendr", "Huxo", "Ramo", "Tiva", "Wex",
                "Jalo", "Esko", "Pirel", "Gondo", "Ulma"]
TEST_FIRST = ["Margaux", "Anaïs", "Salomé", "Imane", "Océane", "Kenza", "Mei", "Aylin", "Romain",
              "Florian", "Adrien", "Ismaël", "Bilal", "Quentin", "Kofi", "Rafael", "Jonas", "Vikram"]
TEST_LAST = ["Delmas", "Brunet", "Charpentier", "Ait-Ali", "Vasseur", "Masson", "Leroux", "Ziani",
             "Ozturk", "Ferreira", "Petit", "Renaud", "Abadie", "Mensah", "Iyer", "Tran"]
TEST_CANDIDATES = ["Lilia", "Adam", "Selma", "Théo", "Ines", "Walid", "Zoé", "Nassim", "Louise", "Idriss"]

POOLS = {
    "train": {"co_head": CO_HEAD, "first": FIRST, "last": LAST, "candidates": CANDIDATES},
    "test": {"co_head": TEST_CO_HEAD, "first": TEST_FIRST, "last": TEST_LAST, "candidates": TEST_CANDIDATES},
}
for _k in ("co_head", "first", "last", "candidates"):
    assert not set(POOLS["train"][_k]) & set(POOLS["test"][_k]), f"train/test pools overlap: {_k}"
assert not any(t.lower().startswith(h.lower()) or h.lower().startswith(t.lower())
               for t in TEST_CO_HEAD for h in CO_HEAD + FR_NAME_WORDS), "company prefixes overlap"

# Sender address domain when the company is omitted from the text (so company is truly null).
GENERIC_DOMAINS = ["gmail.com", "outlook.fr", "hotmail.fr", "yahoo.fr", "outlook.com"]

TIMEZONES_ABROAD = ["America/New_York", "America/Montreal", "Europe/London", "Asia/Kolkata",
                    "Africa/Casablanca", "Africa/Tunis", "Asia/Dubai", "America/Los_Angeles"]


def pick(rng: random.Random, weights: dict):
    keys = list(weights)
    return rng.choices(keys, weights=[weights[k] for k in keys])[0]


def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s.replace("&", " ")).strip("-")


def fictional_company(rng: random.Random, pool: dict, diverse: bool = False) -> dict:
    r = rng.random() if diverse else 1.0
    if r < 0.30:    # multi-word French-style name
        name = rng.choice(FR_PATTERNS).format(w=rng.choice(FR_NAME_WORDS))
        slug = _slug(name)
    elif r < 0.40:  # acronym
        letters = "".join(rng.choice("BCDFGHJKLMNPRSTVWXZ") for _ in range(rng.choice([3, 4])))
        name = letters + rng.choice(["", "", " Group", " France"])
        slug = letters.lower()
    else:           # syllable brand
        base = rng.choice(pool["co_head"]) + rng.choice(CO_TAIL)
        name = base + rng.choice(CO_SUFFIX)
        slug = _slug(base).replace("-", "")
    co = {"name": name, "domain": slug + rng.choice(TLDS), "sector": rng.choice(SECTORS),
          "domain_matches_name": True}
    if diverse and rng.random() < 0.20:  # parent group or ATS relay domain
        co["domain"] = (f"groupe-{_slug(rng.choice(FR_NAME_WORDS))}.fr" if rng.random() < 0.5
                        else rng.choice(RELAY_DOMAINS))
        co["domain_matches_name"] = False
    return co


def send_datetime(rng: random.Random, abroad: bool) -> str:
    day = datetime(2026, 1, 5) + timedelta(days=rng.randrange(0, 340))
    if day.weekday() >= 5 and rng.random() < 0.85:
        day += timedelta(days=7 - day.weekday())  # mostly weekdays
    tz = ZoneInfo(rng.choice(TIMEZONES_ABROAD) if abroad else "Europe/Paris")
    dt = day.replace(hour=rng.randint(7, 21), minute=rng.randint(0, 59), second=rng.randint(0, 59), tzinfo=tz)
    return format_datetime(dt)


def sample_spec(rng: random.Random, spec_id: str, pool: dict) -> dict:
    etype = pick(rng, TYPE_WEIGHTS)
    role = pick(rng, SENDER_ROLE_WEIGHTS[etype])
    subtype = None
    if etype == "other":
        subtype = rng.choice(list(OTHER_SUBTYPES))
        stage = OTHER_SUBTYPES[subtype]
        if subtype in ("job_alert_digest", "job_board_newsletter"):
            role = "automated_system"
    else:
        stage = pick(rng, STAGE_WEIGHTS[etype])

    edges: list[str] = []
    if role == "automated_system":
        edges.append("ats_template")
    allowed = [e for e, types in EDGE_CASES.items()
               if e != "ats_template" and (types is None or etype in types)]
    if role == "automated_system":  # templates aren't informal, agency-written, or forwarded threads
        allowed = [e for e in allowed if e not in ("informal_startup", "agency_confidential_client", "typos",
                                                   "reply_thread", "forwarded_two_companies")]
    if role != "recruiter":  # an agency/ESN sender is a recruiter by definition (rules §7)
        allowed = [e for e in allowed if e != "agency_confidential_client"]
    n_edges = rng.choices([0, 1, 2], weights=[25, 45, 30])[0]
    for _ in range(min(n_edges, len(allowed))):
        e = rng.choices(allowed, weights=[EDGE_WEIGHTS[a] for a in allowed])[0]
        edges.append(e)
        allowed.remove(e)
    if etype == "rejection" and rng.random() < 0.5 and "vague_rejection" not in edges:
        edges.append("vague_rejection")

    language = "mixed_fr_en" if "fr_en_mix" in edges else pick(rng, LANG_WEIGHTS)
    length = pick(rng, LENGTH_WEIGHTS["automated" if role == "automated_system" else "human"])
    register = ("informal_startup" if "informal_startup" in edges
                else "corporate_formal" if role == "automated_system"
                else "terse" if length == "short" and rng.random() < 0.6
                else rng.choice(REGISTERS[:3]))

    mention, omit = [], []
    for field, p in MENTION_PROB.items():
        (mention if rng.random() < p else omit).append(field)
    if "missing_fields" in edges and len(omit) < 2:
        for f in ("job_title", "company"):
            if f in mention:
                mention.remove(f)
                omit.append(f)
                break

    spec = {
        "spec_id": spec_id,
        "email_type": etype,
        "target_stage": stage,
        "language": language,
        "register": register,
        "length": {"bucket": length, "body_words": list(LENGTHS[length])},
        # indirect: fields in `mention` appear only via paraphrase, the subject line, a quoted
        # message or the signature, never as a neat label-like statement.
        "mention_style": "indirect" if rng.random() < 0.5 else "explicit",
        "date_header": send_datetime(rng, abroad="timezone" in edges),
        "sender": {"name": f"{rng.choice(pool['first'])} {rng.choice(pool['last'])}", "role": role},
        "company": fictional_company(rng, pool, diverse=pool is POOLS["train"]),
        "candidate_first_name": rng.choice(pool["candidates"]),
        "role_domain": rng.choice(ROLE_DOMAINS),
        "mention": mention,
        "omit": omit,
        "edge_cases": edges,
    }
    if not spec["company"]["domain_matches_name"] and "company" in omit:
        omit.remove("company")
        mention.append("company")
    # Rule §3: a company domain in From: is enough to extract the company. To get true
    # nulls, half of the company-omitted specs send from a generic mailbox instead.
    # Only senders who plausibly write from a personal mailbox: startup people and terse senders.
    generic = ("company" in omit and role != "automated_system"
               and register in ("informal_startup", "terse") and rng.random() < 0.6)
    spec["sender"]["email_domain"] = rng.choice(GENERIC_DOMAINS) if generic else spec["company"]["domain"]
    if subtype:
        spec["other_subtype"] = subtype
    if "contract_type" in mention:
        spec["contract_type"] = pick(rng, CONTRACT_WEIGHTS)
    if "work_mode" in mention:
        spec["work_mode"] = pick(rng, WORK_MODE_WEIGHTS)
    if "forwarded_two_companies" in edges:
        spec["second_company"] = fictional_company(rng, pool, diverse=pool is POOLS["train"])
    if etype in DEADLINE_PROB:
        spec["action_has_due_date"] = "relative_deadline" in edges or rng.random() < DEADLINE_PROB[etype]
    return spec


# --------------------------------------------------------------------------- validation vs spec

def parse_date_header(email: str):
    m = re.search(r"^Date:\s*(.+)$", email, flags=re.MULTILINE)
    if not m:
        return None, None
    try:
        return m.group(1).strip(), parsedate_to_datetime(m.group(1).strip())
    except (TypeError, ValueError):
        return m.group(1).strip(), None


def check_example(ex: dict, spec: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Errors reject the example."""
    errors, warnings = [], []
    email, label = ex.get("email"), ex.get("label")
    if not isinstance(email, str) or not email.startswith("From:"):
        return ["format: email missing or not starting with From:"], []
    if not isinstance(label, dict):
        return ["format: label is not an object"], []

    raw_date, sent = parse_date_header(email)
    if raw_date != spec["date_header"]:
        errors.append(f"spec: Date header {raw_date!r} != {spec['date_header']!r}")
    errors += validate(label, sent.date() if sent else None)
    if any(e.startswith("schema:") for e in errors):
        return errors, warnings

    if label["email_type"] != spec["email_type"]:
        errors.append(f"spec: email_type {label['email_type']} != {spec['email_type']}")
    if label["sender_role"] != spec["sender"]["role"]:
        errors.append(f"spec: sender_role {label['sender_role']} != {spec['sender']['role']}")
    if label["stage"] != spec["target_stage"]:
        errors.append(f"spec: stage {label['stage']} != target {spec['target_stage']}")
    company_domain = (spec["sender"].get("email_domain", spec["company"]["domain"]) == spec["company"]["domain"]
                      and spec["company"].get("domain_matches_name", True))
    for f in spec["omit"]:
        if label[f] is None:
            continue
        if f == "company" and (company_domain or "forwarded_two_companies" in spec["edge_cases"]):
            warnings.append(f"spec: company omitted from text but extracted ({label[f]!r}) from domain/thread")
        else:
            errors.append(f"spec: {f} should be null (omitted) but is {label[f]!r}")
    for f in spec["mention"]:
        if label[f] is None:
            warnings.append(f"spec: {f} was to be mentioned but label is null")
    if "contract_type" in spec and label["contract_type"] not in (None, spec["contract_type"]):
        warnings.append(f"spec: contract_type {label['contract_type']} != {spec['contract_type']}")
    if spec.get("action_has_due_date") and label["deadline"] is None:
        warnings.append("spec: due date requested but deadline is null")
    lo, hi = spec.get("length", {}).get("body_words", (0, 10**6))
    body = re.split(r"\n\s*\n", email, maxsplit=1)[-1]
    body = re.split(r"\n(?:Le .{5,80} a écrit|On .{5,80} wrote|-{2,}\s*\n|-{3,}|>)", body + "\n")[0]  # own text, no signature/quotes
    n_words = len(body.split())
    if not lo * 0.6 <= n_words <= hi * 1.5:
        warnings.append(f"spec: body has {n_words} words, target {lo}-{hi}")
    if label["next_step"] and re.search(r"[éèàùç]", label["next_step"]):
        warnings.append("rule: next_step looks non-English")
    return errors, warnings


# --------------------------------------------------------------------------- emit / ingest

def build_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8").replace(
        "{LABELING_RULES}", RULES_PATH.read_text(encoding="utf-8"))


def emit(args) -> None:
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)
    pool = POOLS["train" if args.split == "train" else "test"]
    specs = [sample_spec(rng, f"{args.split}-{run_id}-{i:04d}", pool) for i in range(args.n)]
    work = GEN_DIR / args.split / "work"
    (work / "generation_prompt.md").parent.mkdir(parents=True, exist_ok=True)
    (work / "generation_prompt.md").write_text(build_system_prompt(), encoding="utf-8")
    for i in range(0, len(specs), args.batch_size):
        bid = f"{run_id}-b{i // args.batch_size:03d}"
        write_json(work / "pending" / f"{bid}.json",
                   {"split": args.split, "generator": SPLIT_PROFILE[args.split], "run_id": run_id,
                    "seed": seed, "batch_id": bid, "specs": specs[i:i + args.batch_size]})
    print("email_type:", dict(Counter(s["email_type"] for s in specs)))
    print("edge cases:", dict(Counter(e for s in specs for e in s["edge_cases"])))
    print(f"emitted {len(specs)} specs in {-(-len(specs) // args.batch_size)} batches to {work / 'pending'} "
          f"(split={args.split}, generator={SPLIT_PROFILE[args.split]}, seed={seed})")


def canonicalize_company_label(label: dict, spec: dict, email: str) -> dict:
    """If the label names the spec's company (any descriptor variant) and the canonical form is
    written in the email, store that canonical form. Otherwise keep the label as written."""
    lab = label["company"]
    for co in (spec["company"], spec.get("second_company")):
        if lab and co and normalize_text(co["name"]).startswith(normalize_company(lab)):
            canon = canonical_company(co["name"])
            if canon.lower() in email.lower():
                label["company"] = canon
            break
    return label


def ingest_batch(batch: dict, text: str) -> tuple[list[dict], list[dict]]:
    bid, split = batch["batch_id"], batch["split"]
    try:
        examples = parse_json(text)["examples"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return [], [{"batch_id": bid, "errors": [f"unparseable response: {e}"]}]
    by_id = {s["spec_id"]: s for s in batch["specs"]}
    valid, rejected, seen = [], [], set()
    for ex in examples:
        spec = by_id.get(ex.get("spec_id"))
        if spec is None or ex["spec_id"] in seen:
            rejected.append({"batch_id": bid, "example": ex, "errors": ["unknown or duplicate spec_id"]})
            continue
        seen.add(ex["spec_id"])
        errors, warnings = check_example(ex, spec)
        rec = {"id": spec["spec_id"], "split": split, "generator": batch["generator"], "batch_id": bid,
               "run_id": batch["run_id"], "email": ex.get("email"), "label": ex.get("label"),
               "spec": spec, "warnings": warnings}
        if errors:
            rejected.append({**rec, "errors": errors})
        else:
            rec["label"] = canonicalize_company_label(canonical(rec["label"]), spec, rec["email"])
            valid.append(rec)
    for sid in by_id.keys() - seen:
        rejected.append({"batch_id": bid, "id": sid, "errors": ["missing from response"]})
    return valid, rejected


def ingest(args) -> None:
    work = GEN_DIR / args.split / "work"
    prompt = (work / "generation_prompt.md").read_text(encoding="utf-8")
    n_valid = n_total = 0
    for resp_path in sorted((work / "responses").glob("*.json")):
        pending = work / "pending" / resp_path.name
        if not pending.exists():
            print(f"skip {resp_path.name}: no pending batch (already ingested?)")
            continue
        batch = json.loads(pending.read_text(encoding="utf-8"))
        text = resp_path.read_text(encoding="utf-8")
        valid, rejected = ingest_batch(batch, text)
        append_jsonl(GEN_DIR / args.split / "examples.jsonl", valid)
        append_jsonl(GEN_DIR / args.split / "rejected.jsonl", rejected)
        log_usage({"run_id": batch["run_id"], "script": "generate", "split": args.split,
                   "generator": batch["generator"], "batch_id": batch["batch_id"],
                   "n_items": len(batch["specs"]), "n_valid": len(valid),
                   "est_input_tokens": est_tokens(prompt, pending.read_text(encoding="utf-8")),
                   "est_output_tokens": est_tokens(text)})
        for src in (pending, resp_path):  # mark as done
            dst = work / "done" / src.parent.name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
        n_valid += len(valid)
        n_total += len(batch["specs"])
        reasons = Counter(" ".join(e.split()[:2]) for r in rejected for e in r["errors"])
        print(f"[{batch['batch_id']}] {len(valid)}/{len(batch['specs'])} valid"
              + (f" | rejections: {dict(reasons)}" if rejected else ""))
    left = len(list((work / "pending").glob("*.json"))) if (work / "pending").exists() else 0
    print(f"\ningested {n_valid}/{n_total} valid; {left} batches still pending")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["emit", "ingest"])
    ap.add_argument("--split", required=True, choices=list(SPLIT_PROFILE))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    emit(args) if args.mode == "emit" else ingest(args)


if __name__ == "__main__":
    main()

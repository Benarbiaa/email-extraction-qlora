"""Field-level comparison of two labels. Used by relabel agreement and by eval."""

from __future__ import annotations

import re
import unicodedata

LEGAL_SUFFIXES = re.compile(r"\b(sas|sasu|sa|sarl|inc|ltd|llc|gmbh|plc)\b\.?")
# labeling_rules.md §3: generic trailing descriptors are not part of the company name.
DESCRIPTORS = re.compile(r"(\s+(france|group|groupe|systems|technologies|solutions|labs|analytics|"
                         r"digital|consulting|conseil|co|associes))+$")
# Same list on the original text (case and accents kept), for canonical gold labels.
_DESCRIPTORS_RAW = re.compile(r"(\s*(?:,|\b(?:SAS|SASU|SA|SARL|Inc\.?|Ltd|LLC|GmbH|PLC)\b|\bFrance|\bGroup|\bGroupe"
                              r"|\bSystems|\bTechnologies|\bSolutions|\bLabs|\bAnalytics|\bDigital|\bConsulting"
                              r"|\bConseil|&\s*Co\b|&\s*Associés))+\s*$")


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = LEGAL_SUFFIXES.sub(" ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def normalize_company(s: str) -> str:
    core = DESCRIPTORS.sub("", normalize_text(s))
    return core or normalize_text(s)


def canonical_company(name: str) -> str:
    """Core brand per labeling_rules.md §3, keeping the original casing and accents."""
    core = _DESCRIPTORS_RAW.sub("", name).strip()
    return core or name.strip()


def field_match(field: str, gold, pred) -> bool:
    if field == "next_step":  # free text: scored as null/non-null agreement here
        return (gold is None) == (pred is None)
    if field in ("company", "job_title"):
        if gold is None or pred is None:
            return gold is pred
        norm = normalize_company if field == "company" else normalize_text
        return norm(gold) == norm(pred)
    return gold == pred

"""Shared file helpers for generate.py and relabel.py.

Generation runs on a subscription through LLM sub-agents, not an API: the scripts write
batch files to <split>/work/pending/, the sub-agents write <split>/work/responses/, and
the scripts ingest and validate the responses. Committed files only record a generator
letter (A = main generator, B = cross-generator slice), never a model name.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "data" / "generated"
USAGE_PATH = GEN_DIR / "usage.csv"
USAGE_COLS = ["run_id", "script", "split", "generator", "batch_id", "n_items", "n_valid",
              "est_input_tokens", "est_output_tokens", "billing"]
OTHER = {"A": "B", "B": "A"}


def est_tokens(*texts: str) -> int:
    """Rough token estimate (~4 chars/token). Subscription runs expose no exact counts."""
    return sum(len(t) for t in texts) // 4


def log_usage(row: dict) -> None:
    new = not USAGE_PATH.exists()
    with USAGE_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=USAGE_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({"billing": "subscription", **row})


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    # Repair Python-style \UXXXXXXXX escapes (invalid JSON) that writers sometimes emit for emojis.
    text = re.sub(r"(?<!\\)\\U([0-9a-fA-F]{8})", lambda m: chr(int(m.group(1), 16)), text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text[text.find("{"):text.rfind("}") + 1])


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

"""Hand verification of the synthetic test set, then freeze it.

Protocol (README "How the test set was verified"):
  1. blind      Label 50 test emails yourself WITHOUT seeing the gold labels.
  2. agreement  Compare your blind labels with the gold labels (accuracy, Cohen's kappa).
  3. sheet      Review all 200 items with gold + blind-model labels visible, most
                suspicious first; mark ok / corrections.
  4. freeze     Apply your corrections -> data/splits/test.jsonl (+ sha256, corrections log).

Usage:
    python scripts/review.py blind
    python scripts/review.py agreement
    python scripts/review.py sheet
    python scripts/review.py freeze

Filling the CSVs (open in LibreOffice / Google Sheets):
  - empty cell = null; booleans: true / false; dates: YYYY-MM-DD
  - sheet corrections: `field=value; field=value` (value `null` allowed), e.g.
    `stage=screening; deadline=null`
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import GEN_DIR, read_jsonl  # noqa: E402
from src.matching import field_match  # noqa: E402
from src.validation import FIELDS, canonical, validate  # noqa: E402

SLICES = ["test_indist", "test_xgen"]
REVIEW_DIR = ROOT / "data" / "review"
BLIND_CSV = REVIEW_DIR / "blind_labels.csv"
SHEET_CSV = REVIEW_DIR / "test_review.csv"
TABLES = ROOT / "results" / "tables"


def load_slice(name: str) -> list[dict]:
    relabeled = GEN_DIR / name / "relabeled.jsonl"
    return read_jsonl(relabeled if relabeled.exists() else GEN_DIR / name / "examples.jsonl")


def parse_cell(field: str, v: str):
    v = v.strip()
    if v == "" or v.lower() == "null":
        return None
    if field == "action_required":
        if v.lower() not in ("true", "false"):
            raise ValueError(f"action_required must be true/false, got {v!r}")
        return v.lower() == "true"
    return v


def blind(args) -> None:
    if BLIND_CSV.exists():
        raise SystemExit(f"{BLIND_CSV} exists; delete it to re-sample (don't, once you started labeling)")
    rng = random.Random(args.seed)
    rows = []
    for name in SLICES:
        rows += [(name, r) for r in rng.sample(load_slice(name), args.n // len(SLICES))]
    rng.shuffle(rows)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with BLIND_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "email", *FIELDS])
        for _, r in rows:
            w.writerow([r["id"], r["email"], *[""] * len(FIELDS)])
    print(f"wrote {BLIND_CSV} ({len(rows)} emails, no gold labels). Fill the 10 label columns.")


def cohen_kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(args) -> None:
    gold = {r["id"]: (name, r) for name in SLICES for r in load_slice(name)}
    with BLIND_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    filled = [r for r in rows if any(r[k].strip() for k in FIELDS)]
    if not filled:
        raise SystemExit("blind_labels.csv has no filled rows yet")
    human = {r["id"]: {k: parse_cell(k, r[k]) for k in FIELDS} for r in filled}
    for i, lab in human.items():
        if errs := validate(lab):
            print(f"warning: your label for {i} is invalid: {errs}")
    TABLES.mkdir(parents=True, exist_ok=True)
    out = TABLES / "human_vs_gold.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field", "agreement", "cohen_kappa", "n"])
        print(f"human (blind) vs gold on {len(human)} emails")
        for k in FIELDS:
            hv = [human[i][k] for i in human]
            gv = [gold[i][1]["label"][k] for i in human]
            acc = sum(field_match(k, g, h) for g, h in zip(gv, hv)) / len(hv)
            kappa = f"{cohen_kappa([str(x) for x in hv], [str(x) for x in gv]):.3f}" if k in ("email_type", "stage", "sender_role") else ""
            w.writerow([k, f"{acc:.3f}", kappa, len(hv)])
            print(f"  {k:16} {acc:6.1%}" + (f"  kappa={kappa}" if kappa else ""))
    print(f"wrote {out}")


def sheet(args) -> None:
    if SHEET_CSV.exists() and not args.force:
        raise SystemExit(f"{SHEET_CSV} exists (it may contain your review). Use --force to overwrite.")
    rows = [(name, r) for name in SLICES for r in load_slice(name)]
    rows.sort(key=lambda x: (-len(x[1].get("disagreements", [])), x[0], x[1]["id"]))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    with SHEET_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "slice", "n_flags", "flagged_fields", "email", "gold_label", "blind_model_label",
                    "ok", "corrections", "notes"])
        for name, r in rows:
            w.writerow([r["id"], name, len(r.get("disagreements", [])), " ".join(r.get("disagreements", [])),
                        r["email"], json.dumps(r["label"], ensure_ascii=False),
                        json.dumps(r.get("relabel"), ensure_ascii=False), "", "", ""])
    print(f"wrote {SHEET_CSV} ({len(rows)} items, flagged first). For each row: ok=y, or corrections.")


def freeze(args) -> None:
    with SHEET_CSV.open(encoding="utf-8") as f:
        review = {r["id"]: r for r in csv.DictReader(f)}
    todo = [i for i, r in review.items() if not r["ok"].strip() and not r["corrections"].strip()]
    if todo:
        raise SystemExit(f"{len(todo)} items not reviewed yet (empty ok and corrections), e.g. {todo[:3]}")
    out_rows, corrections = [], []
    for name in SLICES:
        for r in load_slice(name):
            label = dict(r["label"])
            rev = review[r["id"]]
            for part in filter(None, (p.strip() for p in rev["corrections"].split(";"))):
                field, _, value = part.partition("=")
                field = field.strip()
                if field not in FIELDS:
                    raise SystemExit(f"{r['id']}: unknown field in correction {part!r}")
                new = parse_cell(field, value)
                corrections.append([r["id"], name, field, json.dumps(label[field], ensure_ascii=False),
                                    json.dumps(new, ensure_ascii=False), rev["notes"]])
                label[field] = new
            if errs := validate(label):
                raise SystemExit(f"{r['id']}: corrected label is invalid: {errs}")
            out_rows.append({"id": r["id"], "slice": name, "email": r["email"], "label": canonical(label)})
    split = ROOT / "data" / "splits" / "test.jsonl"
    split.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows)
    split.write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode()).hexdigest()
    (split.parent / "test.sha256").write_text(f"{digest}  test.jsonl\n")
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "test_corrections.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "slice", "field", "before", "after", "notes"])
        w.writerows(corrections)
    by_field = Counter(c[2] for c in corrections)
    print(f"froze {len(out_rows)} items -> {split} (sha256 {digest[:12]}…)")
    print(f"{len(corrections)} corrections in {len({c[0] for c in corrections})} items; by field: {dict(by_field)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["blind", "agreement", "sheet", "freeze"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    {"blind": blind, "agreement": agreement, "sheet": sheet, "freeze": freeze}[args.mode](args)


if __name__ == "__main__":
    main()

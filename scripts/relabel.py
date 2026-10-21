"""Blind relabeling: a second model labels each generated email from the text + rules only.

The labeler never sees the spec or the generator's label: its batch files contain only
ids and email texts. It is the *other* generator (A <-> B). Disagreements are review
signals: this script never drops or edits anything.

Two steps (see scripts/_common.py):
    emit    examples.jsonl -> <split>/work/relabel_pending/rNNN.json + the rendered prompt
    ingest  <split>/work/relabel_responses/rNNN.json -> relabeled.jsonl + agreement table

Usage:
    python scripts/relabel.py emit --split train
    python scripts/relabel.py ingest --split train
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import GEN_DIR, OTHER, est_tokens, log_usage, parse_json, read_jsonl, write_json  # noqa: E402
from scripts.generate import parse_date_header  # noqa: E402
from src.matching import field_match  # noqa: E402
from src.validation import FIELDS, canonical, validate  # noqa: E402

PROMPT_PATH = ROOT / "data" / "prompts" / "relabel_system.md"
RULES_PATH = ROOT / "data" / "labeling_rules.md"


def labeler_for(records: list[dict]) -> str:
    generators = sorted({r["generator"] for r in records})
    if len(generators) != 1:
        raise SystemExit(f"Mixed generators {generators} in one split; relabel them separately")
    return OTHER[generators[0]]


def emit(args) -> None:
    records = read_jsonl(GEN_DIR / args.split / "examples.jsonl")
    labeler = labeler_for(records)
    work = GEN_DIR / args.split / "work"
    (work / "relabel_prompt.md").write_text(PROMPT_PATH.read_text(encoding="utf-8").replace(
        "{LABELING_RULES}", RULES_PATH.read_text(encoding="utf-8")), encoding="utf-8")
    for i in range(0, len(records), args.batch_size):
        write_json(work / "relabel_pending" / f"r{i // args.batch_size:03d}.json",
                   {"labeler": labeler,
                    "emails": [{"id": r["id"], "email": r["email"]} for r in records[i:i + args.batch_size]]})
    print(f"emitted {len(records)} emails in {-(-len(records) // args.batch_size)} batches "
          f"to {work / 'relabel_pending'} (labeler={labeler})")


def compare(rec: dict, pred) -> list[str]:
    """Fields where the blind label disagrees with the gold label. Invalid label -> all fields."""
    _, sent = parse_date_header(rec["email"])
    if not isinstance(pred, dict) or validate(pred, sent.date() if sent else None):
        return list(FIELDS)
    return [f for f in FIELDS if not field_match(f, rec["label"][f], pred[f])]


def ingest(args) -> None:
    split_dir = GEN_DIR / args.split
    work = split_dir / "work"
    records = read_jsonl(split_dir / "examples.jsonl")
    labeler = labeler_for(records)
    prompt = (work / "relabel_prompt.md").read_text(encoding="utf-8")

    # Keep blind labels from earlier (incremental) relabel runs; new responses override them.
    preds = {}
    if (split_dir / "relabeled.jsonl").exists():
        preds = {r["id"]: r["relabel"] for r in read_jsonl(split_dir / "relabeled.jsonl")
                 if r.get("relabel_by") == labeler and r.get("relabel") is not None}
    for resp in sorted((work / "relabel_responses").glob("*.json")):
        pending = work / "relabel_pending" / resp.name
        text = resp.read_text(encoding="utf-8")
        try:
            labels = {x["id"]: x["label"] for x in parse_json(text)["labels"]}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"{resp.name}: unparseable ({e})")
            labels = {}
        preds.update(labels)
        n_items = len(json.loads(pending.read_text(encoding="utf-8"))["emails"]) if pending.exists() else 0
        log_usage({"run_id": records[0]["run_id"], "script": "relabel", "split": args.split,
                   "generator": labeler, "batch_id": resp.stem, "n_items": n_items, "n_valid": len(labels),
                   "est_input_tokens": est_tokens(prompt, pending.read_text(encoding="utf-8")) if pending.exists() else 0,
                   "est_output_tokens": est_tokens(text)})

    out_rows, disagree = [], Counter()
    confusions = {"email_type": Counter(), "stage": Counter()}
    for r in records:
        pred = preds.get(r["id"])
        d = compare(r, pred)
        disagree.update(d)
        for f in confusions:
            if f in d and isinstance(pred, dict) and f in pred:
                confusions[f][(r["label"][f], pred[f])] += 1
        relabel = canonical(pred) if isinstance(pred, dict) and set(FIELDS) <= pred.keys() else pred
        out_rows.append({**r, "relabel": relabel, "relabel_by": labeler, "disagreements": d})

    with (split_dir / "relabeled.jsonl").open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(records)
    full = sum(not row["disagreements"] for row in out_rows)
    table = ROOT / "results" / "tables" / f"label_agreement_{args.split}.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["field", "agreement", "n_disagree", "n"])
        for fld in FIELDS:
            w.writerow([fld, f"{1 - disagree[fld] / n:.3f}", disagree[fld], n])
        w.writerow(["ALL_FIELDS", f"{full / n:.3f}", n - full, n])

    print(f"labeler={labeler} on {args.split}: {len(preds)}/{n} labeled")
    print(f"{'field':16} agreement")
    for fld in FIELDS:
        print(f"{fld:16} {1 - disagree[fld] / n:6.1%}  ({disagree[fld]} disagree)")
    print(f"{'ALL FIELDS':16} {full / n:6.1%}")
    for fld, c in confusions.items():
        if c:
            print(f"\ntop {fld} disagreements (gold -> blind):")
            for (g, p), k in c.most_common(5):
                print(f"  {g} -> {p}: {k}")
    print(f"\nwrote {split_dir / 'relabeled.jsonl'} and {table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["emit", "ingest"])
    ap.add_argument("--split", required=True)
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()
    emit(args) if args.mode == "emit" else ingest(args)


if __name__ == "__main__":
    main()

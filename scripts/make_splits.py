"""Build train/val/test splits with near-duplicate filtering.

- Near-duplicates: character 5-gram Jaccard on the email body (headers dropped,
  lowercased, whitespace collapsed).
  * train vs test: drop any train email with similarity >= --test-threshold to a test email.
  * within train: drop the later email of any pair with similarity >= --train-threshold.
- Validation: whole generation batches (never split a batch), seeded, until >= --val-size.
- Test: test_indist + test_xgen. Until the hand review is frozen (scripts/review.py freeze ->
  data/splits/test.jsonl), it is written as test_provisional.jsonl with verified=false.

Usage:
    python scripts/make_splits.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._common import GEN_DIR, read_jsonl  # noqa: E402

SPLITS = ROOT / "data" / "splits"
TABLES = ROOT / "results" / "tables"


def body(email: str) -> str:
    text = re.split(r"\n\s*\n", email, maxsplit=1)[-1]
    return re.sub(r"\s+", " ", text.lower()).strip()


def shingles(text: str, k: int = 5) -> set[str]:
    return {text[i:i + k] for i in range(max(len(text) - k + 1, 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def slim(r: dict, **extra) -> dict:
    """Fields the training / eval code needs. The spec stays only as analysis tags."""
    return {"id": r["id"], "email": r["email"], "label": r["label"], "batch_id": r["batch_id"],
            "generator": r["generator"], "edge_cases": r["spec"]["edge_cases"],
            "language": r["spec"]["language"], "length": r["spec"]["length"]["bucket"]
            if "length" in r["spec"] else None, **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-size", type=int, default=100)
    ap.add_argument("--test-threshold", type=float, default=0.5)
    ap.add_argument("--train-threshold", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    train = read_jsonl(GEN_DIR / "train" / "examples.jsonl")
    test = [(s, r) for s in ("test_indist", "test_xgen") for r in read_jsonl(GEN_DIR / s / "examples.jsonl")]
    tr_sh = {r["id"]: shingles(body(r["email"])) for r in train}
    te_sh = [shingles(body(r["email"])) for _, r in test]

    # train vs test leakage
    max_sim = {r["id"]: max(jaccard(tr_sh[r["id"]], t) for t in te_sh) for r in train}
    leaked = {i for i, s in max_sim.items() if s >= args.test_threshold}

    # within-train near-duplicates (keep the first, drop later ones)
    dup = set()
    ids = [r["id"] for r in train]
    for i, a in enumerate(ids):
        if a in dup or a in leaked:
            continue
        for b in ids[i + 1:]:
            if b not in dup and jaccard(tr_sh[a], tr_sh[b]) >= args.train_threshold:
                dup.add(b)
    kept = [r for r in train if r["id"] not in leaked | dup]

    # company-name overlap check (pools are disjoint by construction; verify it)
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower()) if s else None  # noqa: E731
    train_cos = {norm(r["label"]["company"]) for r in kept} - {None}
    test_cos = {norm(r["label"]["company"]) for _, r in test} - {None}
    overlap = sorted(train_cos & test_cos)

    # validation = whole batches
    rng = random.Random(args.seed)
    batches = sorted({r["batch_id"] for r in kept})
    rng.shuffle(batches)
    val_batches, n = set(), 0
    by_batch = Counter(r["batch_id"] for r in kept)
    for b in batches:
        if n >= args.val_size:
            break
        val_batches.add(b)
        n += by_batch[b]
    tr = [slim(r) for r in kept if r["batch_id"] not in val_batches]
    va = [slim(r) for r in kept if r["batch_id"] in val_batches]
    te = [slim(r, slice=s, verified=False) for s, r in test]

    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", tr), ("val", va), ("test_provisional", te)):
        with (SPLITS / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    TABLES.mkdir(parents=True, exist_ok=True)
    sims = sorted(max_sim.values())
    q = lambda p: sims[min(int(p * len(sims)), len(sims) - 1)]  # noqa: E731
    with (TABLES / "leakage_check.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in [("train_generated", len(train)), ("dropped_test_near_dup", len(leaked)),
                     ("dropped_train_near_dup", len(dup)), ("train", len(tr)), ("val", len(va)),
                     ("test", len(te)), ("max_sim_train_to_test_p50", f"{q(.5):.3f}"),
                     ("max_sim_train_to_test_p95", f"{q(.95):.3f}"), ("max_sim_train_to_test_max", f"{sims[-1]:.3f}"),
                     ("company_name_overlap_train_test", len(overlap))]:
            w.writerow([k, v])

    print(f"train generated {len(train)} | dropped: {len(leaked)} near-dup of test (>= {args.test_threshold}), "
          f"{len(dup)} near-dup within train (>= {args.train_threshold})")
    print(f"splits: train {len(tr)} | val {len(va)} ({len(val_batches)} whole batches) | test {len(te)} (provisional)")
    print(f"train->test nearest-neighbour Jaccard: p50 {q(.5):.3f}, p95 {q(.95):.3f}, max {sims[-1]:.3f}")
    print(f"company names shared by train and test: {len(overlap)} {overlap[:10]}")
    print("email_type train:", dict(Counter(r["label"]["email_type"] for r in tr).most_common()))
    print("email_type val:  ", dict(Counter(r["label"]["email_type"] for r in va).most_common()))


if __name__ == "__main__":
    main()

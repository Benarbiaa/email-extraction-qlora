"""Evaluate predictions against gold labels.

Metrics (README "Evaluation"):
  valid_json        output parses as a JSON object
  schema_valid      ... and passes data/schema.json
  rule_consistent   ... and passes the cross-field labeling rules (email_type x stage, ...)
  per-field accuracy (all fields; next_step = null/non-null agreement) and macro-F1 for enum fields
  next_step_token_f1  token F1 on items where both gold and prediction are non-null
  exact_match       all fields except next_step correct
  latency           mean / p50 / p95 seconds per email (from the predictions file)
Invalid outputs count as wrong on every field.

Usage:
    python src/eval.py --pred results/preds/base_zeroshot.jsonl --gold data/splits/test_provisional.jsonl --name base_zeroshot
Predictions JSONL: {"id": ..., "output": "<raw model text>", "latency_s": 1.23}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.matching import field_match  # noqa: E402
from src.validation import FIELDS, rule_errors, schema_errors  # noqa: E402

ENUM_FIELDS = ["email_type", "contract_type", "work_mode", "stage", "sender_role", "action_required"]
EM_FIELDS = [f for f in FIELDS if f != "next_step"]
INVALID = "<invalid>"


def parse_output(text: str):
    """Extract the first JSON object from model output; None if impossible."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def token_f1(a: str, b: str) -> float:
    ta, tb = re.findall(r"\w+", a.lower()), re.findall(r"\w+", b.lower())
    common = sum((Counter(ta) & Counter(tb)).values())
    if not ta or not tb or not common:
        return 0.0
    p, r = common / len(tb), common / len(ta)
    return 2 * p * r / (p + r)


def macro_f1(gold: list, pred: list) -> float:
    labels = {str(g) for g in gold}
    f1s = []
    for lab in labels:
        tp = sum(str(g) == lab and str(p) == lab for g, p in zip(gold, pred))
        fp = sum(str(g) != lab and str(p) == lab for g, p in zip(gold, pred))
        fn = sum(str(g) == lab and str(p) != lab for g, p in zip(gold, pred))
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def score(rows: list[tuple[dict, dict | None]]) -> dict:
    """rows: (gold record, parsed prediction or None). Returns a flat metrics dict."""
    n = len(rows)
    m = {"n": n}
    parsed = [p for _, p in rows]
    schema_ok = [p is not None and not schema_errors(p) for p in parsed]
    m["valid_json"] = sum(p is not None for p in parsed) / n
    m["schema_valid"] = sum(schema_ok) / n
    m["rule_consistent"] = sum(ok and not rule_errors(p) for ok, p in zip(schema_ok, parsed)) / n
    usable = [p if ok else None for ok, p in zip(schema_ok, parsed)]

    per_field_ok = defaultdict(list)
    for (g, _), p in zip(rows, usable):
        for f in FIELDS:
            per_field_ok[f].append(p is not None and field_match(f, g["label"][f], p[f]))
    for f in FIELDS:
        m[f"acc_{f}"] = sum(per_field_ok[f]) / n
    for f in ENUM_FIELDS:
        m[f"f1_{f}"] = macro_f1([g["label"][f] for g, _ in rows],
                                [p[f] if p is not None else INVALID for p in usable])
    both = [(g["label"]["next_step"], p["next_step"]) for (g, _), p in zip(rows, usable)
            if p is not None and g["label"]["next_step"] and p["next_step"]]
    m["next_step_token_f1"] = sum(token_f1(a, b) for a, b in both) / len(both) if both else 0.0
    m["exact_match"] = sum(all(per_field_ok[f][i] for f in EM_FIELDS) for i in range(n)) / n
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "tables"))
    args = ap.parse_args()

    gold = {r["id"]: r for r in map(json.loads, open(args.gold, encoding="utf-8"))}
    preds = {r["id"]: r for r in map(json.loads, open(args.pred, encoding="utf-8"))}
    missing = gold.keys() - preds.keys()
    if missing:
        print(f"warning: {len(missing)} gold items have no prediction; they count as invalid")
    rows = [(g, parse_output(preds[i]["output"]) if i in preds else None) for i, g in gold.items()]

    report = {"name": args.name, "gold": Path(args.gold).name,
              "verified_gold": all(g.get("verified", True) for g in gold.values()),
              "overall": score(rows)}
    lat = [preds[i]["latency_s"] for i in gold if i in preds and "latency_s" in preds[i]]
    if lat:
        lat.sort()
        report["overall"].update(latency_mean_s=statistics.mean(lat), latency_p50_s=lat[len(lat) // 2],
                                 latency_p95_s=lat[min(int(0.95 * len(lat)), len(lat) - 1)])
    for key in ("slice", "language", "length"):
        groups = defaultdict(list)
        for g, p in rows:
            if g.get(key):
                groups[g[key]].append((g, p))
        if groups:
            report[f"by_{key}"] = {k: score(v) for k, v in sorted(groups.items())}
    edge = defaultdict(list)
    for g, p in rows:
        for e in g.get("edge_cases", []) or ["(none)"]:
            edge[e].append((g, p))
    report["by_edge_case"] = {k: score(v) for k, v in sorted(edge.items())}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"metrics_{args.name}.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    summary = out / "summary.csv"
    cols = ["name", "gold", "verified_gold", "n", "valid_json", "schema_valid", "rule_consistent", "exact_match",
            *[f"acc_{f}" for f in FIELDS], "next_step_token_f1", "latency_mean_s"]
    existing = list(csv.DictReader(open(summary, encoding="utf-8"))) if summary.exists() else []
    existing = [r for r in existing if not (r["name"] == args.name and r["gold"] == report["gold"])]
    o = report["overall"]
    existing.append({"name": args.name, "gold": report["gold"], "verified_gold": report["verified_gold"],
                     **{c: (f"{o[c]:.4f}" if isinstance(o.get(c), float) else o.get(c, "")) for c in cols[3:]}})
    with summary.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(existing)

    tag = "" if report["verified_gold"] else "  [PROVISIONAL: gold not hand-verified]"
    print(f"{args.name} on {report['gold']}{tag}")
    for k in ["valid_json", "schema_valid", "rule_consistent", "exact_match"] + [f"acc_{f}" for f in FIELDS]:
        print(f"  {k:22} {o[k]:.1%}")
    if lat:
        print(f"  latency mean {o['latency_mean_s']:.2f}s  p95 {o['latency_p95_s']:.2f}s")
    if "by_slice" in report:
        print("  exact_match by slice:", {k: f"{v['exact_match']:.1%}" for k, v in report["by_slice"].items()})


if __name__ == "__main__":
    main()

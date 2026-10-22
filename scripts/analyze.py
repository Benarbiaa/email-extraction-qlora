"""Plots, confidence intervals and error analysis from the prediction files (no GPU needed).

Reads results/preds/*.jsonl + the gold split and writes:
  results/plots/field_accuracy.png       per-field accuracy + exact match, all models
  results/plots/exact_match_breakdown.png exact match by slice, language and length
  results/plots/loss_curve.png           train / val loss of the fine-tuning run
  results/tables/exact_match_ci.csv      bootstrap 95% CIs + paired difference vs the fine-tuned model
  results/error_analysis/<model>_errors.csv  one row per wrong field (gold vs prediction)
  results/error_analysis/summary.md      top confusions per model, out-of-schema values

Usage:
    python scripts/analyze.py            # needs matplotlib
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import EM_FIELDS, parse_output  # noqa: E402
from src.matching import field_match  # noqa: E402
from src.validation import FIELDS, schema_errors  # noqa: E402

PREDS, TABLES = ROOT / "results" / "preds", ROOT / "results" / "tables"
PLOTS, ERRORS = ROOT / "results" / "plots", ROOT / "results" / "error_analysis"
MODELS = {"base_zeroshot": "Base zero-shot", "base_fewshot": "Base few-shot", "ft_qlora_v1": "Fine-tuned (QLoRA v1)"}
FT = "ft_qlora_v1"
COLORS = {"base_zeroshot": "#9aa5b1", "base_fewshot": "#5b7083", "ft_qlora_v1": "#d9822b"}


def gold_path() -> Path:
    frozen = ROOT / "data" / "splits" / "test.jsonl"
    return frozen if frozen.exists() else ROOT / "data" / "splits" / "test_provisional.jsonl"


def load(name: str, gold: dict) -> dict:
    """id -> (parsed prediction or None if invalid, list of wrong fields)."""
    preds = {r["id"]: r["output"] for r in map(json.loads, open(PREDS / f"{name}.jsonl", encoding="utf-8"))}
    out = {}
    for i, g in gold.items():
        p = parse_output(preds.get(i, ""))
        if p is None or schema_errors(p):
            out[i] = (p, list(FIELDS))
        else:
            out[i] = (p, [f for f in FIELDS if not field_match(f, g["label"][f], p[f])])
    return out


def exact(res: dict) -> dict:
    return {i: not any(f in EM_FIELDS for f in bad) for i, (_, bad) in res.items()}


def bootstrap(em: dict, ids: list, ref: dict | None = None, n: int = 10000, seed: int = 0):
    rng = random.Random(seed)
    stats = []
    for _ in range(n):
        s = [ids[rng.randrange(len(ids))] for _ in ids]
        v = sum(em[i] for i in s) / len(s)
        stats.append(v - sum(ref[i] for i in s) / len(s) if ref else v)
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n)]


def plots(gold: dict, results: dict, ems: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS.mkdir(parents=True, exist_ok=True)
    n = len(gold)
    cols = ["exact_match"] + list(FIELDS)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    w = 0.27
    for k, name in enumerate(results):
        vals = [sum(ems[name].values()) / n] + [sum(f not in bad for _, bad in results[name].values()) / n
                                                 for f in FIELDS]
        ax.bar([x + (k - 1) * w for x in range(len(cols))], vals, w, label=MODELS[name], color=COLORS[name])
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(f"accuracy (n={n})")
    ax.set_title("Per-field accuracy and full-record exact match (test set, provisional labels)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "field_accuracy.png", dpi=150)
    plt.close(fig)

    groups = [("slice", "slice"), ("language", "language"), ("length", "length")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (key, title) in zip(axes, groups):
        cats = sorted({g[key] for g in gold.values()})
        for k, name in enumerate(results):
            vals = []
            for c in cats:
                ids = [i for i, g in gold.items() if g[key] == c]
                vals.append(sum(ems[name][i] for i in ids) / len(ids))
            ax.bar([x + (k - 1) * w for x in range(len(cats))], vals, w, label=MODELS[name], color=COLORS[name])
        counts = Counter(g[key] for g in gold.values())
        ax.set_xticks(range(len(cats)), [f"{c}\n(n={counts[c]})" for c in cats])
        ax.set_title(f"by {title}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("exact match")
    axes[0].set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(PLOTS / "exact_match_breakdown.png", dpi=150)
    plt.close(fig)

    log = list(csv.DictReader(open(TABLES / "train_log_qlora_v1.csv", encoding="utf-8")))
    tr = [(int(r["step"]), float(r["train_loss"])) for r in log if r["train_loss"]]
    ev = [(int(r["step"]), float(r["eval_loss"])) for r in log if r["eval_loss"]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(*zip(*tr), alpha=0.6, label="train loss", color="#5b7083")
    ax.plot(*zip(*ev), marker="o", label="val loss", color="#d9822b")
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step (16 examples / step, 52 steps / epoch)")
    ax.set_ylabel("loss (completion tokens)")
    ax.set_title("QLoRA v1 training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "loss_curve.png", dpi=150)
    plt.close(fig)


def main():
    gpath = gold_path()
    gold = {r["id"]: r for r in map(json.loads, open(gpath, encoding="utf-8"))}
    results = {m: load(m, gold) for m in MODELS if (PREDS / f"{m}.jsonl").exists()}
    ems = {m: exact(r) for m, r in results.items()}
    ids = list(gold)

    with (TABLES / "exact_match_ci.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "gold", "n", "exact_match", "ci95_low", "ci95_high", "diff_vs_ft", "diff_ci95_low",
                    "diff_ci95_high"])
        for m in results:
            em = sum(ems[m].values()) / len(ids)
            lo, hi = bootstrap(ems[m], ids)
            row = [m, gpath.name, len(ids), f"{em:.4f}", f"{lo:.4f}", f"{hi:.4f}"]
            if m != FT and FT in ems:
                dlo, dhi = bootstrap(ems[m], ids, ref=ems[FT])
                row += [f"{em - sum(ems[FT].values()) / len(ids):.4f}", f"{dlo:.4f}", f"{dhi:.4f}"]
            w.writerow(row)
            print(f"{m:15} exact match {em:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")

    ERRORS.mkdir(parents=True, exist_ok=True)
    md = [f"# Error analysis ({gpath.name}{', provisional labels' if 'provisional' in gpath.name else ''})", ""]
    for m, res in results.items():
        rows, conf, oos = [], Counter(), Counter()
        for i, (p, bad) in res.items():
            g = gold[i]
            if p is not None and schema_errors(p):
                for e in schema_errors(p):
                    oos[e.split(" is not")[0].replace("schema:", "")] += 1
            for f in bad:
                pv = "<invalid>" if p is None or schema_errors(p) else p[f]
                rows.append([i, g["slice"], g["language"], g["label"]["email_type"], f,
                             json.dumps(g["label"][f], ensure_ascii=False), json.dumps(pv, ensure_ascii=False)])
                if pv != "<invalid>":
                    gv = g["label"][f]
                    if f in ("company", "job_title", "deadline", "next_step"):
                        key = f"{'null' if gv is None else 'value'} -> {'null' if pv is None else 'value'}" \
                            if (gv is None) != (pv is None) else "wrong value"
                    else:
                        key = f"{gv} -> {pv}"
                    conf[(f, key)] += 1
        with (ERRORS / f"{m}_errors.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "slice", "language", "gold_email_type", "field", "gold", "pred"])
            w.writerows(rows)
        n_bad = sum(any(x in EM_FIELDS for x in bad) for _, bad in res.values())
        md += [f"## {MODELS[m]}", "",
               f"- records with at least one wrong scored field: {n_bad}/{len(res)}",
               f"- out-of-schema outputs: {sum(p is None or bool(schema_errors(p)) for p, _ in res.values())}", ""]
        if oos:
            md += ["Out-of-schema values:", ""] + [f"- `{k}` ×{v}" for k, v in oos.most_common(8)] + [""]
        md += ["| field | gold → predicted | count |", "|---|---|---|"]
        md += [f"| {f} | {k} | {v} |" for (f, k), v in conf.most_common(15)] + [""]
    (ERRORS / "summary.md").write_text("\n".join(md), encoding="utf-8")

    plots(gold, results, ems)
    print(f"wrote plots to {PLOTS}, error analysis to {ERRORS}, CIs to {TABLES / 'exact_match_ci.csv'}")


if __name__ == "__main__":
    main()

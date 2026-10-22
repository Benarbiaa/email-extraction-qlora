"""Publish the LoRA adapter + a model card to the Hugging Face Hub.

Reads HF_TOKEN from the environment or .env. Metrics in the card come from
results/tables/summary.csv and exact_match_ci.csv, so card and repo cannot disagree.

    python scripts/push_to_hub.py --adapter checkpoints/qlora_v1/final --repo <user>/<name> [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "results" / "tables"
GITHUB = "https://github.com/Benarbiaa/email-extraction-qlora"
UPLOAD = ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json",
          "chat_template.jinja"]  # not training_args.bin (a pickle, and not needed to load the adapter)


def token() -> str:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    env = ROOT / ".env"
    m = re.search(r"^HF_TOKEN=(.+)$", env.read_text(encoding="utf-8"), re.M) if env.exists() else None
    if not m or not m.group(1).strip():
        raise SystemExit("HF_TOKEN not set (environment or .env)")
    return m.group(1).strip()


def pct(x: str) -> str:
    return f"{float(x):.1%}"


def model_card(repo: str) -> str:
    rows = {(r["name"], r["gold"]): r for r in csv.DictReader(open(TABLES / "summary.csv", encoding="utf-8"))}
    ci = {r["model"]: r for r in csv.DictReader(open(TABLES / "exact_match_ci.csv", encoding="utf-8"))}
    gold = next(g for n, g in rows if n == "ft_qlora_v1")
    zs, fs, ft = (rows[(n, gold)] for n in ("base_zeroshot", "base_fewshot", "ft_qlora_v1"))
    provisional = "provisional" in gold

    def line(label, key):
        return f"| {label} | {pct(zs[key])} | {pct(fs[key])} | **{pct(ft[key])}** |"

    em = lambda m: f"{pct(ci[m]['exact_match'])} [{pct(ci[m]['ci95_low'])}, {pct(ci[m]['ci95_high'])}]"  # noqa: E731
    return f"""---
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: peft
license: apache-2.0
language: [fr, en]
pipeline_tag: text-generation
tags: [lora, qlora, information-extraction, json, recruitment, email]
---

# Recruitment email → JSON (Qwen3-4B-Instruct-2507, QLoRA)

LoRA adapter that turns a recruitment/HR email (French, English or mixed) into a JSON record
with 10 fields: `email_type`, `company`, `job_title`, `contract_type`, `work_mode`, `stage`,
`sender_role`, `action_required`, `deadline`, `next_step`.
Schema, labeling rules, data pipeline, training and evaluation code: **[{GITHUB}]({GITHUB})**.

> Trained and evaluated on **synthetic** emails only (fictional companies and people).
{"> Test labels are **provisional** (automatic validation + blind cross-model relabeling; hand review pending)." if provisional else ""}

## Results (200 test emails, 4-bit base, greedy decoding)

| | Base zero-shot | Base few-shot | **This adapter** |
|---|---|---|---|
| Exact match, 9 fields (95% CI) | {em("base_zeroshot")} | {em("base_fewshot")} | **{em("ft_qlora_v1")}** |
{line("Schema-valid", "schema_valid")}
{line("email_type", "acc_email_type")}
{line("stage", "acc_stage")}
{line("deadline", "acc_deadline")}
| Mean latency (T4) | {float(zs["latency_mean_s"]):.2f} s | {float(fs["latency_mean_s"]):.2f} s | {float(ft["latency_mean_s"]):.2f} s |

Half the test set is written by a different generator model than the training data; on that half
the adapter reaches 80% exact match. Baselines get the labeling rules as a written prompt; the
adapter uses a one-line instruction. Details, plots and error analysis are in the GitHub README.

## Usage

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base = "Qwen/Qwen3-4B-Instruct-2507"
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
model = PeftModel.from_pretrained(model, "{repo}")

email = open("email.txt").read()   # include the From:/Date:/Subject: headers
messages = [{{"role": "system", "content": "Extract the recruitment email record as JSON (schema v1)."}},
            {{"role": "user", "content": "Email:\\n" + email}}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The `Date:` header is used to resolve relative deadlines ("avant vendredi"). Validate the output
against the schema (`data/schema.json` in the GitHub repo) before using it.

## Training

QLoRA: 4-bit nf4 base (double quantization, fp16 compute), LoRA r=16, alpha=32, dropout 0.05 on all
linear layers; 824 training emails, 2 epochs (104 steps, effective batch 16), lr 2e-4 cosine,
loss on the JSON completion only, max length 1024 (no example truncated). Checkpoint selected by
validation loss (105 emails). One free Colab T4.

## Limitations

- Synthetic data only; accuracy on real emails is unknown.
- Labels follow this project's own conventions (e.g. company = core brand name, first call =
  `screening`); see the labeling rules in the GitHub repo.
- Personal project for learning and demonstration, not for automated decisions about candidates.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "checkpoints" / "qlora_v1" / "final"))
    ap.add_argument("--repo", required=True, help="<user>/<model-name>")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the card to /tmp and stop")
    args = ap.parse_args()

    adapter = Path(args.adapter)
    missing = [f for f in UPLOAD if not (adapter / f).exists()]
    if missing:
        raise SystemExit(f"missing in {adapter}: {missing}")
    card = model_card(args.repo)
    if args.dry_run:
        Path("/tmp/model_card.md").write_text(card, encoding="utf-8")
        print(card)
        return

    from huggingface_hub import HfApi
    api = HfApi(token=token())
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo, folder_path=str(adapter), allow_patterns=UPLOAD,
                      commit_message="Upload QLoRA adapter (qlora_v1)")
    api.upload_file(repo_id=args.repo, path_or_fileobj=card.encode("utf-8"), path_in_repo="README.md",
                    commit_message="Model card")
    print(f"published https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    sys.exit(main())

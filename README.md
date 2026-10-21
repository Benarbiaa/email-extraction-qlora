# email-extraction-qlora

Fine-tuning **Qwen3-4B-Instruct-2507** with QLoRA to turn recruitment/HR emails (FR/EN/mixed)
into a schema-validated JSON record, evaluated against the same base model (zero-shot and
few-shot) on a hand-verified test set.

**Model choice.** Qwen3-4B-Instruct-2507 (Apache 2.0, text-only, no thinking mode) over the
originally planned Qwen2.5-3B (older generation) and the newest Qwen3.5-4B / Gemma 4 E4B (hybrid
linear-attention or multimodal architectures with thinner tooling support on a free T4).
Fallback: Qwen3-1.7B.

**Prompts.** Baselines get a condensed version of the labeling rules (~520 tokens) as the system
prompt, plus 3 fixed train examples for few-shot. The fine-tuned model is trained and run with
a one-line instruction: with the full prompt, ~20% of training examples would exceed the
1024-token budget. Each model is evaluated in its intended configuration; part of the
fine-tuned model's latency gain comes from its shorter prompt.

> **Status:** work in progress (Day 1).

## Data disclosure
Training data is **synthetic** (LLM-generated, fictional companies and people).
The 200-item test set is hand-verified (procedure: TBD, documented here on Day 2).
A real-world test split (own anonymized emails) is out of scope for now.

## Layout
- `data/` — schema, labeling rules, generation prompts, splits
- `src/` — `train.py`, `eval.py`, `infer.py`
- `scripts/` — data generation & validation
- `notebooks/` — baseline, training, eval
- `app/` — demo
- `results/` — tables, plots, error analysis

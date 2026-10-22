"""QLoRA supervised fine-tuning on the train split (validation = val split; test is never read).

- Base model in 4-bit nf4 (double quant, fp16 compute: the T4 has no bf16).
- LoRA r=16, alpha=32, dropout 0.05 on all linear layers.
- Conversational prompt-completion format (prompting.training_example): one-line system
  prompt + email -> canonical JSON. Loss on the completion tokens only.
- No truncation: every example is tokenized first and training refuses to start if one
  exceeds --max-length (a truncated target would teach the model broken JSON).
- Checkpoints every --save-steps to --output-dir (point it at Drive on Colab); rerunning
  the same command resumes from the last checkpoint. The best checkpoint by val loss is
  saved as <output-dir>/final.

Usage (GPU):
    python src/train.py --output-dir /content/drive/MyDrive/email-extraction-qlora/runs/qlora_v1
    python src/train.py --output-dir /tmp/smoke --limit 32 --max-steps 4   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.infer import DEFAULT_MODEL  # noqa: E402
from src.prompting import training_example  # noqa: E402

SPLITS = ROOT / "data" / "splits"


def read_split(path: Path, limit: int | None = None) -> list[dict]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return rows[:limit]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def token_lengths(tok, examples: list[dict]) -> list[int]:
    """Full sequence length (prompt + completion) as the trainer will see it."""
    texts = [tok.apply_chat_template(ex["prompt"] + ex["completion"], tokenize=False) for ex in examples]
    return [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]


def check_lengths(tok, name: str, examples: list[dict], records: list[dict], max_length: int) -> None:
    lengths = token_lengths(tok, examples)
    too_long = [(r["id"], n) for r, n in zip(records, lengths) if n > max_length]
    srt = sorted(lengths)
    print(f"{name}: {len(lengths)} examples, tokens p50 {srt[len(srt) // 2]}, "
          f"p95 {srt[min(int(0.95 * len(srt)), len(srt) - 1)]}, max {srt[-1]} (limit {max_length})")
    if too_long:
        raise SystemExit(f"{len(too_long)} {name} examples exceed {max_length} tokens and would be truncated: "
                         f"{too_long[:5]}. Raise --max-length or drop them in make_splits.py.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--train", default=str(SPLITS / "train.jsonl"))
    ap.add_argument("--val", default=str(SPLITS / "val.jsonl"))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--run-name", default="qlora_v1")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N train/val rows (smoke test)")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (smoke test)")
    args = ap.parse_args()
    if "test" in Path(args.train).name or "test" in Path(args.val).name:
        raise SystemExit("refusing to train or select checkpoints on a test split")

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from transformers.trainer_utils import get_last_checkpoint
    from trl import SFTConfig, SFTTrainer

    train_rows, val_rows = read_split(Path(args.train), args.limit), read_split(Path(args.val), args.limit)
    train_ex, val_ex = [training_example(r) for r in train_rows], [training_example(r) for r in val_rows]

    tok = AutoTokenizer.from_pretrained(args.model)
    check_lengths(tok, "train", train_ex, train_rows, args.max_length)
    check_lengths(tok, "val", val_ex, val_rows, args.max_length)

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map="auto",
                                                 dtype=torch.float16)
    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                      target_modules="all-linear", task_type="CAUSAL_LM")

    out = Path(args.output_dir)
    config = SFTConfig(
        output_dir=str(out), run_name=args.run_name, seed=args.seed,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_steps=5, weight_decay=0.0, optim="paged_adamw_8bit",
        fp16=True, bf16=False, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length, packing=False, completion_only_loss=True,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1, report_to="none",
    )
    trainer = SFTTrainer(model=model, args=config, processing_class=tok, peft_config=lora,
                         train_dataset=Dataset.from_list(train_ex), eval_dataset=Dataset.from_list(val_ex))
    trainer.model.print_trainable_parameters()

    last = get_last_checkpoint(str(out)) if out.exists() else None
    if last:
        print(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)

    final = out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    best = trainer.state.best_model_checkpoint
    print(f"best checkpoint {best} (eval_loss {trainer.state.best_metric:.4f}) saved to {final}")

    # Run record: config + data hashes + loss curves (for the README and plots).
    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    record = {"run_name": args.run_name, "base_model": args.model, "args": vars(args),
              "train_sha256": sha256(Path(args.train)), "val_sha256": sha256(Path(args.val)),
              "n_train": len(train_rows), "n_val": len(val_rows), "best_checkpoint": best,
              "best_eval_loss": trainer.state.best_metric, "global_steps": trainer.state.global_step,
              "versions": {"torch": torch.__version__, **{m: __import__(m).__version__
                                                          for m in ("transformers", "peft", "trl", "bitsandbytes")}}}
    (tables / f"train_{args.run_name}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    with (tables / f"train_log_{args.run_name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "epoch", "train_loss", "eval_loss", "learning_rate"])
        for h in trainer.state.log_history:
            if "loss" in h or "eval_loss" in h:
                w.writerow([h.get("step"), h.get("epoch"), h.get("loss", ""), h.get("eval_loss", ""),
                            h.get("learning_rate", "")])
    print(f"wrote {tables / f'train_{args.run_name}.json'} and train_log_{args.run_name}.csv")


if __name__ == "__main__":
    main()

"""Run a model on a split and write raw predictions + latency (input for src/eval.py).

Modes:
  zeroshot   base model, full instruction (prompting.SYSTEM_PROMPT)
  fewshot    base model, full instruction + 3 fixed train examples
  finetuned  base model + LoRA adapter, one-line instruction (prompting.FINETUNE_SYSTEM_PROMPT)
All modes load the base model in 4-bit by default, so the comparison is quantization-matched.
Decoding is greedy; thinking is disabled for models that support it (Qwen3 hybrids).

Usage (GPU):
    python src/infer.py --mode zeroshot --split data/splits/test_provisional.jsonl --out results/preds/base_zeroshot.jsonl
    python src/infer.py --mode finetuned --adapter <dir or hub id> --split ... --out results/preds/ft_v1.jsonl
The output file is appended to and already-predicted ids are skipped, so a Colab disconnect
only loses the email in progress.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.prompting import FINETUNE_SYSTEM_PROMPT, SYSTEM_PROMPT, chat_messages, select_few_shot  # noqa: E402

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def load(model_id: str = DEFAULT_MODEL, adapter: str | None = None, four_bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16) if four_bit else None
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map="auto",
                                                 dtype=torch.float16)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def render(tok, messages: list[dict]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def generate(tok, model, messages: list[dict], max_new_tokens: int = 256) -> tuple[str, float]:
    import torch

    inputs = tok(render(tok, messages), return_tensors="pt").to(model.device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True), latency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["zeroshot", "fewshot", "finetuned"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", required=True)
    ap.add_argument("--train", default=str(ROOT / "data" / "splits" / "train.jsonl"), help="few-shot source")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()
    if args.mode == "finetuned" and not args.adapter:
        raise SystemExit("--mode finetuned needs --adapter")

    rows = [json.loads(line) for line in open(args.split, encoding="utf-8")][: args.limit]
    few_shot = select_few_shot([json.loads(line) for line in open(args.train, encoding="utf-8")]) \
        if args.mode == "fewshot" else None
    system = FINETUNE_SYSTEM_PROMPT if args.mode == "finetuned" else SYSTEM_PROMPT
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if out.exists() else set()

    tok, model = load(args.model, args.adapter, four_bit=not args.no_4bit)
    todo = [r for r in rows if r["id"] not in done]
    prompt_tokens = len(tok(render(tok, chat_messages(rows[0]["email"], few_shot, system)))["input_ids"])
    print(f"{args.mode} with {args.model}: {len(todo)} to predict ({len(done)} already in {out}); "
          f"first prompt = {prompt_tokens} tokens")
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            text, latency = generate(tok, model, chat_messages(r["email"], few_shot, system), args.max_new_tokens)
            f.write(json.dumps({"id": r["id"], "output": text, "latency_s": round(latency, 3)},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  last latency {latency:.2f}s", flush=True)


if __name__ == "__main__":
    main()

"""Kaggle runner: clone the repo, train the QLoRA adapter, evaluate on val then test.

Pushed as a Kaggle script kernel (see kaggle/README.md); everything it produces lands in
/kaggle/working and is fetched with `kaggle kernels output`.
STAGE = "smoke": 2 training steps on 32 examples + 3 predictions (checks the stack, ~10 min).
STAGE = "full":  full training run + val and test predictions and metrics.
"""

import os
import shutil
import subprocess
import sys

STAGE = "smoke"
RUN = "qlora_v1"
REPO_URL = "https://github.com/Benarbiaa/email-extraction-qlora.git"
REPO = "/kaggle/working/repo"
OUT = "/kaggle/working"

# One T4 is enough for a 4-bit 4B model; two would make Trainer wrap the model in DataParallel.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=REPO if os.path.exists(REPO) else OUT)


sh(f"git clone -q {REPO_URL} {REPO}")
sh("git log --oneline -1")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
sh(f"{sys.executable} -m pip install -q -U transformers peft trl bitsandbytes accelerate datasets jsonschema")
sh(f"{sys.executable} -c \"import torch, transformers, peft, trl, bitsandbytes; "
   "print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, "
   "'| trl', trl.__version__, '| bnb', bitsandbytes.__version__)\"")

py = f"{sys.executable} -u"
test = "data/splits/test.jsonl" if os.path.exists(f"{REPO}/data/splits/test.jsonl") else "data/splits/test_provisional.jsonl"
run_dir = f"{OUT}/runs/{RUN}"

if STAGE == "smoke":
    sh(f"{py} src/train.py --output-dir /tmp/smoke --run-name smoke --limit 32 --max-steps 2 --eval-steps 1 --save-steps 2")
    sh(f"{py} src/infer.py --mode finetuned --adapter /tmp/smoke/final --split data/splits/val.jsonl --limit 3 "
       f"--out {OUT}/smoke_preds.jsonl")
    sh(f"cat {OUT}/smoke_preds.jsonl")
else:
    sh(f"{py} src/train.py --output-dir {run_dir} --run-name {RUN}")
    sh(f"{py} src/infer.py --mode finetuned --adapter {run_dir}/final --split data/splits/val.jsonl "
       f"--out results/preds/ft_{RUN}_val.jsonl")
    sh(f"{py} src/eval.py --pred results/preds/ft_{RUN}_val.jsonl --gold data/splits/val.jsonl --name ft_{RUN}_val")
    sh(f"{py} src/infer.py --mode finetuned --adapter {run_dir}/final --split {test} --out results/preds/ft_{RUN}.jsonl")
    sh(f"{py} src/eval.py --pred results/preds/ft_{RUN}.jsonl --gold {test} --name ft_{RUN}")
    # Keep only the selected adapter; intermediate checkpoints (with optimizer state) are large.
    for d in os.listdir(run_dir):
        if d.startswith("checkpoint-"):
            shutil.rmtree(f"{run_dir}/{d}")

shutil.copytree(f"{REPO}/results", f"{OUT}/results", dirs_exist_ok=True)
shutil.rmtree(REPO)
print("done", flush=True)

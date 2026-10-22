# Running on Kaggle (free T4)

`run_kaggle.py` is pushed as a private Kaggle script kernel. It clones this repo from GitHub,
so commit and push before running. Requires a phone-verified Kaggle account (GPU + internet).

```bash
pip install kaggle                     # token in ~/.kaggle/access_token
# set STAGE = "smoke" or "full" in run_kaggle.py
kaggle kernels push -p kaggle          # starts a run
kaggle kernels status emirbenarbia/email-extraction-qlora
kaggle kernels logs emirbenarbia/email-extraction-qlora
kaggle kernels output emirbenarbia/email-extraction-qlora -p /tmp/kaggle_out
```

Outputs: `results/` (predictions, metrics, training log) and `runs/<run>/final` (the LoRA adapter).
Colab alternative: `notebooks/02_train.ipynb`.

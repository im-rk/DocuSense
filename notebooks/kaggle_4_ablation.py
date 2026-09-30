# %% [markdown]
# # DocuSense - Notebook 4 (optional): ablation study for the report
# **Accelerator: GPU. Internet: OFF.** **Add Data**: `docusense-code` + `docusense-data`.
# Each variant trains ~40 epochs (about 10-25 min). Run a few variants per session (or all, ~3 h).
# Output: `ablation.csv` - the measured table for the report.

# %% Cell 1 - setup
import glob, os, sys, json, csv
import torch
hits = sorted(glob.glob("/kaggle/input/**/models/tagger/labels.py", recursive=True), key=len)   # any nesting depth
if not hits:   # show what IS attached, so the problem is visible
    for dp, ds, fs in os.walk("/kaggle/input"):
        if dp.count("/") <= 5: print(dp, ds[:6], fs[:4])

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))))
found = glob.glob("/kaggle/input/**/cord_train.jsonl", recursive=True)
DATA = os.path.dirname(found[0])
OUT = "/kaggle/working/ablation"; os.makedirs(OUT, exist_ok=True)
from models.tagger.train import train_tagger

# %% Cell 2 - variants (edit SEEDS to [42, 43, 44] for mean +- std if you have GPU quota left)
SEEDS = [42]
EPOCHS = 40
REAL = ("cord_train", "sroie_train")
VARIANTS = {
    "full (text+look+pos, BiLSTM x2)": dict(),
    "text only":                       dict(use_look=False, use_pos=False),
    "text + position":                 dict(use_look=False),
    "text + look":                     dict(use_pos=False),
    "no context (no BiLSTM)":          dict(use_lstm=False),
    "BiLSTM x1":                       dict(lstm_layers=1),
    "naive CE (no partial-label loss)": dict(partial_loss=False),
    "no class weights":                dict(class_weights=False),
    "no augmentation":                 dict(augment=False),
    "no synthetic data":               dict(train_sets=REAL),
}

# %% Cell 3 - run (per-variant results are appended to ablation.csv so a crash loses nothing)
COLS = ["variant", "seed", "macro_f1", "params", "acc_vendor", "acc_date", "acc_total", "acc_subtotal", "acc_tax"]
csv_path = f"{OUT}/ablation.csv"
if not os.path.exists(csv_path):
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=COLS).writeheader()
for i, (name, kw) in enumerate(VARIANTS.items()):
    for seed in SEEDS:
        tag = f"abl{i}_s{seed}"
        m = train_tagger(DATA, OUT, epochs=EPOCHS, patience=8, seed=seed, tag=tag, num_workers=2, resume=False, **kw)
        t = m["test"]["ALL_TEST"]
        row = {"variant": name, "seed": seed, "macro_f1": t["tokens"]["macro_f1"], "params": m["params"],
               **{f"acc_{k}": v["acc"] for k, v in t["fields"].items()}}
        print("RESULT", row)
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore", restval="").writerow(row)
        for p in glob.glob(f"{OUT}/{tag}_*.pth"):                    # keep the output folder small
            os.remove(p)

# %% Cell 4 - table
import pandas as pd
df = pd.read_csv(csv_path)
print(df.groupby("variant")[["macro_f1"] + [c for c in df.columns if c.startswith("acc_")]].mean().round(4).to_string())
# Download ablation.csv from the Output tab.

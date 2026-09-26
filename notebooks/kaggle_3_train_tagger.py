# %% [markdown]
# # DocuSense - Notebook 3: train Model 2 (word tagger: three-branch CNN + BiLSTM, from scratch)
# **Accelerator: GPU T4 x2 (or P100). Internet: OFF is fine.**
# **Add Data**: `docusense-code` + `docusense-data`.  Runtime ~1 h.
# Outputs to download afterwards: `tagger_v1.pth` (contains weights + config + character vocabulary),
# `tagger_report.json`, `tagger_curves.png`.

# %% Cell 1 - setup
import glob, os, sys, json
import torch
hits = glob.glob("/kaggle/input/*/models/__init__.py") + glob.glob("/kaggle/input/*/*/models/__init__.py")
assert hits, "Attach your `docusense-code` dataset"
sys.path.insert(0, os.path.dirname(os.path.dirname(hits[0])))
found = glob.glob("/kaggle/input/*/cord_train.jsonl") + glob.glob("/kaggle/input/*/*/cord_train.jsonl")
assert found, "Attach your `docusense-data` dataset (built by notebook 1)"
DATA = os.path.dirname(found[0])
OUT = "/kaggle/working/ckpt"; os.makedirs(OUT, exist_ok=True)
print("data:", DATA, "| gpu:", torch.cuda.is_available() and torch.cuda.get_device_name(0))
print(sorted(f for f in os.listdir(DATA) if f.endswith(".jsonl")))

# %% Cell 2 - sanity checks (must pass before the long run). ~1-2 min
from models.tagger.train import sanity_overfit
sanity_overfit(DATA, n_docs=8, steps=250)      # expect: params ~2M, initial loss ~ln(10)=2.3, final token accuracy ~1.0

# %% Cell 3 - baseline: rule-based extractor (lower bound for the report)
from models.tagger.baseline import baseline_preds
from models.tagger.dataset import load_records
from models.tagger.evaluate import field_accuracy, meter_from_preds
allp = []
for name in ("cord_test", "sroie_test"):
    recs, _ = load_records(DATA, name)
    allp += baseline_preds(recs)
print("BASELINE (rules)  macro-F1:", meter_from_preds(allp).report()["macro_f1"], " fields:", field_accuracy(allp))

# %% Cell 4 - FULL TRAINING (per-epoch checkpoint; if the session dies just re-run this cell and it resumes)
from models.tagger.train import train_tagger
metrics = train_tagger(
    DATA, OUT,
    epochs=80, batch_size=8, lr=1e-3, weight_decay=1e-2, patience=12, dropout=0.3,
    d=256, hid=192, lstm_layers=2,
    train_sets=("cord_train", "sroie_train", "synth_train"), val_sets=("cord_val", "sroie_val"),
    test_sets=("cord_test", "sroie_test"), synth_frac=0.3,
    partial_loss=True, class_weights=True, augment=True,
    seed=42, num_workers=2, resume=True, tag="tagger")

# %% Cell 5 - results table (paste these MEASURED numbers into the report / README)
t = metrics["test"]["ALL_TEST"]
print("TEST macro-F1 (token level):", t["tokens"]["macro_f1"])
for k, v in t["tokens"].items():
    if k != "macro_f1": print(f"  {k:11s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")
print("FIELD accuracy:", t["fields"])
for name in ("cord_test", "sroie_test"):
    print(name, metrics["test"][name]["tokens"]["macro_f1"], metrics["test"][name]["fields"])

# %% Cell 6 - curves
from PIL import Image
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4)); plt.imshow(Image.open(f"{OUT}/tagger_curves.png")); plt.axis("off"); plt.show()

# %% Cell 7 - what to download
print(os.listdir(OUT))
# Save Version -> Save & Run All (Commit). From the Output tab download `tagger_v1.pth`
# and put it at   Docusense-dev/models/checkpoints/tagger_v1.pth      (the vocabulary is stored inside it)

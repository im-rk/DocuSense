# %% [markdown]
# # DocuSense - Notebook 2: train Model 1 (page classifier, ResNet from scratch)
# **Accelerator: GPU T4 x2 (or P100). Internet: OFF is fine.**
# **Add Data**: `docusense-code` + `docusense-data`.  Runtime ~30-60 min.
# Outputs to download afterwards: `classifier_v1.pth` (+ curves/confusion PNGs and `classifier_report.json`).

# %% Cell 1 - setup
import glob, os, sys, json
import torch
hits = sorted(glob.glob("/kaggle/input/**/models/tagger/labels.py", recursive=True), key=len)   # any nesting depth
if not hits:   # show what IS attached, so the problem is visible
    for dp, ds, fs in os.walk("/kaggle/input"):
        if dp.count("/") <= 5: print(dp, ds[:6], fs[:4])

assert hits, "Attach your `docusense-code` dataset"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(hits[0]))))
stats = glob.glob("/kaggle/input/**/cls_stats.json", recursive=True)
assert stats, "Attach your `docusense-data` dataset (built by notebook 1)"
DATA = os.path.dirname(stats[0])
OUT = "/kaggle/working/ckpt"; os.makedirs(OUT, exist_ok=True)
print("data:", DATA, "| gpu:", torch.cuda.is_available() and torch.cuda.get_device_name(0))
print(json.load(open(f"{DATA}/cls_stats.json"))["counts"])

# %% Cell 2 - sanity check: the model must be able to memorise 64 images (proves the code is correct). ~1 min
from models.classifier.train import sanity_overfit
sanity_overfit(DATA, n=64, steps=120)          # expect: initial loss ~1.6-2.0, final accuracy ~1.0

# %% Cell 3 - FULL TRAINING (per-epoch checkpoint; if the session dies just re-run this cell and it resumes)
from models.classifier.train import train_classifier
metrics = train_classifier(
    DATA, OUT,
    epochs=30, batch_size=64, lr=2e-3, weight_decay=1e-2,
    arch="resnet", augment=True, seed=42, num_workers=4, resume=True, tag="classifier")
print(json.dumps(metrics, indent=1))            # target: test_acc >= 0.93 (measured, not promised)

# %% Cell 4 - look at the curves and confusion matrix
from PIL import Image
import matplotlib.pyplot as plt
for f in ("classifier_curves.png", "classifier_confusion.png"):
    plt.figure(figsize=(9, 4)); plt.imshow(Image.open(f"{OUT}/{f}")); plt.axis("off"); plt.show()

# %% Cell 5 - OPTIONAL ablations for the report (each ~15 min). Uncomment to run.
# for tag, kw in [("abl_plain_cnn", dict(arch="plain")), ("abl_no_aug", dict(augment=False))]:
#     m = train_classifier(DATA, OUT, epochs=15, num_workers=4, resume=False, tag=tag, **kw)
#     print(tag, m["test_acc"])
# (BatchNorm on/off: edit models/classifier/architecture.py locally and re-bundle if you want that extra row.)

# %% Cell 6 - what to download
print(os.listdir(OUT))
# Save Version -> Save & Run All (Commit). From the Output tab download `classifier_v1.pth`
# and put it at   Docusense-dev/models/checkpoints/classifier_v1.pth

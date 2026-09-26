# %% [markdown]
# # DocuSense - Notebook 1: build the datasets
# **Accelerator: None (CPU)** - saves GPU quota. **Internet: ON.**
# **Add Data**: your `docusense-code` dataset + the Kaggle dataset **SROIE datasetv2** (search "SROIE datasetv2").
# Runtime ~30-60 min. When done: **Save Version -> Save & Run All (Commit)**, then Output tab ->
# **New Dataset** -> name it `docusense-data`. Notebooks 2-4 attach that dataset.

# %% Cell 1 - setup (finds your code dataset automatically)
import glob, os, sys, random, json
import numpy as np

hits = glob.glob("/kaggle/input/*/models/__init__.py") + glob.glob("/kaggle/input/*/*/models/__init__.py")
assert hits, "Code dataset not found: add your `docusense-code` dataset via Add Data (see KAGGLE_TRAINING.md)"
CODE = os.path.dirname(os.path.dirname(hits[0]))
sys.path.insert(0, CODE)
print("code from:", CODE)

os.system("pip install -q datasets")            # Hugging Face `datasets` (data only - no models)
random.seed(42); np.random.seed(42)
WORK = "/kaggle/working/data"
os.makedirs(WORK, exist_ok=True)

# %% Cell 2 - CORD-v2 (receipts, word-level labels) from Hugging Face
from data.build_cord import build_cord
cord_stats = build_cord(WORK, hf_id="naver-clova-ix/cord-v2", audit=20)

# %% Cell 3 - SROIE (vendor / date / total labels). Auto-detects the Kaggle dataset folder
from data.build_sroie import build_sroie
cands = [p for p in glob.glob("/kaggle/input/*") if "sroie" in p.lower()]
assert cands, "SROIE dataset not attached: Add Data -> search 'SROIE datasetv2'"
SROIE_ROOT = cands[0]
print("SROIE root:", SROIE_ROOT)
sroie_res = build_sroie(WORK, SROIE_ROOT, audit=20)
# GATE: coverage must be >= 95%. If not, look at sroie_res['dropped'] and the audit images below, then fix data/build_sroie.py

# %% Cell 4 - LABEL-QUALITY GATE: look at the auto-labelled pages (boxes are coloured by label)
import matplotlib.pyplot as plt
from PIL import Image
imgs = sorted(glob.glob(f"{WORK}/audit/*.jpg"))
pick = imgs[:4] + imgs[len(imgs)//2: len(imgs)//2 + 4]
fig, axes = plt.subplots(2, 4, figsize=(20, 12))
for ax, p in zip(axes.ravel(), pick):
    ax.imshow(Image.open(p)); ax.set_title(os.path.basename(p), fontsize=8); ax.axis("off")
plt.tight_layout(); plt.show()
# Check: vendor (red), date (blue), items (green/orange/purple), subtotal/tax/total sit on the right words.

# %% Cell 5 - synthetic invoices (recommended: adds INVOICE_NO and invoice-style layouts). ~5-10 min
from data.synth_invoices import build_synth
build_synth(WORK, n_train=2000, n_val=200, audit=10)

# %% Cell 6 - Model 1 data: RVL-CDIP streamed with a per-class cap + RECEIPT class from CORD/SROIE. ~15-30 min
from data.build_rvl_cache import build_cls_cache
# Tries these Hugging Face copies in order (classes are mapped by NAME, so different id orders don't matter).
# If they all fail (id/format changed), attach a local RVL-CDIP copy (one sub-folder per class, e.g. invoice/, form/, letter/,
# email/, memo/, resume/, news_article/) and pass rvl_dir="/kaggle/input/<that-dataset>" instead.
stats = build_cls_cache(WORK, quotas={"train": 3000, "val": 500, "test": 500}, hf_ids=("chainyo/rvl-cdip", "aharley/rvl_cdip"))

# %% Cell 7 - summary
import subprocess
print(subprocess.run(["bash", "-c", f"ls -lh {WORK} | head -60; du -sh {WORK}"], capture_output=True, text=True).stdout)
print(json.load(open(f"{WORK}/cls_stats.json")))
# Now: Save Version -> Save & Run All (Commit).  Then Output tab -> New Dataset -> `docusense-data`.

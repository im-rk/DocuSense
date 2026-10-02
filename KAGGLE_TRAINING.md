# Training on Kaggle — step by step

All training (both models) happens on **Kaggle's free GPU**. Nothing is trained on your PC. The flow:

```
your PC ──(1) bundle code──► Kaggle Dataset "docusense-code"
Kaggle Notebook 1 (CPU)  ──► builds data ──► Kaggle Dataset "docusense-data"
Kaggle Notebook 2 (GPU)  ──► trains Model 1  ──► classifier_v1.pth ─┐
Kaggle Notebook 3 (GPU)  ──► trains Model 2  ──► tagger_v1.pth ─────┤ download
                                                                    ▼
                                   Docusense-dev/models/checkpoints/   ← the app loads them from here
```

**Rules that are already built into the code:** no pretrained weights anywhere, random (He/Xavier) initialisation, our own
training loops, per-epoch checkpoints (a dead session can resume), fixed seeds.

---

## 0. One-time Kaggle setup
1. Create/log in to an account at kaggle.com.
2. **Verify your phone number** (Account → Settings → Phone verification). Without it you get **no GPU and no Internet**.
3. Free limits: ~30 GPU-hours/week, 12 h per session, ~20 GB output, ~16 GB RAM. This project needs roughly 2–4 GPU-hours total
   (ablations extra).

## 1. Upload the code as a Kaggle Dataset
On your PC, in the repo root:
```powershell
.venv\Scripts\python.exe tools\make_kaggle_bundle.py
```
This creates `kaggle_bundle/docusense-code/` (folders `models/` and `data/`, code only).
Then on Kaggle: **Datasets → New Dataset** → drag **the two folders inside** `docusense-code` (`models`, `data`) →
Title: **docusense-code** → Visibility: **Private** → Create.

> Re-run the bundler and upload a **new version** of this dataset whenever you change code in `models/` or `data/`.

## 2. Notebook 1 — build the datasets (≈ 30–60 min, no GPU)
- **New Notebook → File → Import Notebook →** `notebooks/kaggle_1_build_data.ipynb` (already generated in the repo).
- **Settings (right panel):** Accelerator **None** · Internet **On** · Persistence: Files only.
- **Add Data:** (a) your **docusense-code**, (b) search **"SROIE datasetv2"** (by *urbikn*) and add it.
- **Run All.** Read the output of cell 3 — **label coverage must be ≥ 95 %** — and look at the pictures in cell 4:
  vendor / date / items / total boxes must sit on the right words. If not, stop and tell me; do not train on bad labels.
- **Save Version → Save & Run All (Commit).** When it finishes: open the version's **Output** tab → **New Dataset** →
  name it **docusense-data** (Private). (≈ 2–3 GB.)

What it builds: CORD-v2 + SROIE word-level labels, word-crop cache, 2 000 synthetic invoices, and the Model-1 image cache
(RVL-CDIP is *streamed* with a per-class cap — never the full 38 GB; classes are matched by name).

## 3. Notebook 2 — train Model 1 (page classifier, ResNet) (≈ 30–60 min)
- Import `notebooks/kaggle_2_train_classifier.ipynb`.
- **Settings:** Accelerator **GPU T4 x2** (or P100) · Internet **Off** is fine.
- **Add Data:** **docusense-code** and **docusense-data**.
- **Run All.** Cell 2 is a code-correctness check (must reach ≈100 % on 64 images). Cell 3 trains 30 epochs.
- Watch: `val_acc` should rise steadily. Final print shows `test_acc` and per-class recall.
  (Target ≥ 0.93 — a target, not a promise; report whatever you measure.)
- **Save Version → Save & Run All**, then download from the **Output** tab: **`classifier_v1.pth`**
  (also `classifier_curves.png`, `classifier_confusion.png`, `classifier_report.json` for the report).

**The training code itself** (if you want to read or copy it): `models/classifier/train.py` (loop),
`architecture.py`, `dataset.py`, `evaluate.py`. The notebook only calls `train_classifier(...)` with these settings:
`epochs=30, batch_size=64, lr=2e-3, weight_decay=1e-2, arch="resnet", augment=True, seed=42`.

## 4. Notebook 3 — train Model 2 (word tagger: three-branch CNN + BiLSTM) (≈ 1 h)
- Import `notebooks/kaggle_3_train_tagger.ipynb`.
- **Settings:** Accelerator **GPU T4 x2** (or P100) · Internet **Off**.
- **Add Data:** **docusense-code** and **docusense-data**.
- **Run All.** Cell 2: sanity check (≈ 2 M parameters, memorises 8 documents). Cell 3: rule-based baseline (lower bound).
  Cell 4: full training (≤ 80 epochs, early stop). Cell 5 prints the **results table** (per-class P/R/F1 + field accuracy).
- **Save Version → Save & Run All**, then download **`tagger_v1.pth`** (the character vocabulary is stored inside it)
  and `tagger_report.json`, `tagger_curves.png`.

**The training code itself:** `models/tagger/train.py` (loop), `architecture.py`, `dataset.py`, `loss.py`
(partial-label loss), `evaluate.py`. Settings used: `epochs=80, batch_size=8, lr=1e-3, weight_decay=1e-2, patience=12,
dropout=0.3, d=256, hid=192, lstm_layers=2, synth_frac=0.3, partial_loss=True, class_weights=True, augment=True, seed=42`.

## 5. (Optional) Notebook 4 — ablation study for the report
`notebooks/kaggle_4_ablation.ipynb` trains 10 variants (text only, +position, +look, no BiLSTM, naive loss, …) and writes
`ablation.csv`. Same settings as notebook 3. Run a few variants per session if quota is tight.

## 6. Put the models into the project
Copy the two downloaded files here (they are git-ignored):
```
Docusense-dev/models/checkpoints/classifier_v1.pth
Docusense-dev/models/checkpoints/tagger_v1.pth
```
Check: start the API (`uvicorn api.main:app --reload`) and open http://localhost:8000/health → `"missing": []`.
Or from a terminal: `python -m pipeline.run path\to\receipt.jpg`.

---

## Settings at a glance
| Notebook | Accelerator | Internet | Add Data | Time | You download |
|---|---|---|---|---|---|
| 1 build data | None | **On** | docusense-code, SROIE datasetv2 | 30–60 min | (save output as dataset **docusense-data**) |
| 2 classifier | GPU | Off | docusense-code, docusense-data | 30–60 min | `classifier_v1.pth` |
| 3 tagger | GPU | Off | docusense-code, docusense-data | ~1 h | `tagger_v1.pth` |
| 4 ablation | GPU | Off | docusense-code, docusense-data | 1–3 h | `ablation.csv` |

## If something goes wrong
| Problem | Fix |
|---|---|
| "Code dataset not found" | Attach **docusense-code** (Add Data). The folders `models/` and `data/` must be inside it. |
| SROIE not found / coverage < 95 % | Attach *SROIE datasetv2*; look at `sroie_res['dropped']` and the audit images; tell me the folder layout you see. |
| `UnidentifiedImageError` / `read operation timed out` in cell 6 | Fixed in the current bundle: corrupt RVL images are skipped, a dropped stream keeps what it collected, and a re-run skips splits already built. Re-bundle and upload a **new version** of docusense-code, then re-run notebook 1. |
| RVL-CDIP streaming fails | Cell 6 streams `chainyo/rvl-cdip` (its validation split is called `val`; handled). If it fails, attach any RVL-CDIP copy with one sub-folder per class and call `build_cls_cache(WORK, rvl_dir="/kaggle/input/<name>")`. |
| Session died / timed out | Re-run the training cell — it **resumes** from the last per-epoch checkpoint in `/kaggle/working/ckpt` (use *Save Version* so it isn't lost). |
| CUDA out of memory | Classifier: `batch_size=32`. Tagger: `batch_size=4`. |
| Metrics far below targets | Don't add layers first. Look at the audit images, the per-class F1 table, and the confusion matrix, then tell me the numbers — the plan is error analysis → fix labels/rules/augmentation → retrain. |
| `Cell 2 sanity check` doesn't reach ≈ 100 % | A code bug — stop and send me the output. |

## What to record for your report (all printed/saved by the notebooks)
Parameter counts · training curves · Model-1 confusion matrix + per-class recall · Model-2 per-class P/R/F1 ·
field-level accuracy (vendor/date/total/subtotal/tax) · rule-baseline vs model · ablation table · then run
`python -m evaluation.run_evaluation` on 20–30 of your own documents for the **real-world** (live OCR) number.

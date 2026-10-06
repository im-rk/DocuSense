# DocuSense — Tester's Guide

How to set up, run and check this project end to end. Follow it top to bottom; every step says what you should see.
Plan on **about 45 minutes** (30 of them are one-time installs).

---

## 0. What you are testing (read this first)

DocuSense reads a photo/scan of a **receipt or invoice** and returns structured data (vendor, date, subtotal, tax, total, invoice number, line items) after checking the arithmetic.

```
upload ─► Model 1: page classifier (ResNet)   RECEIPT | INVOICE | FORM | LETTER | OTHER
       ─► OCR (Tesseract)                      words + boxes   (the only non-from-scratch part)
       ─► Model 2: word tagger (3-branch CNN + BiLSTM)   labels every word: VENDOR, DATE, TOTAL, ITEM_* ...
       ─► post-processing + validation        fields, line items, "subtotal + tax = total" checks
       ─► FastAPI + SQLite ─► React UI        review, correct, export
```

Both neural networks were **trained from scratch on Kaggle** (no pretrained weights). The measured results on held-out test data:

| | Result |
|---|---|
| Model 1 — page classifier, 3.3 M params | **89.5 %** test accuracy |
| Model 2 — word tagger, 1.8 M params | token **macro-F1 0.970** (rule-based baseline: 0.267) |
| Model 2 — field accuracy | date 100 % · subtotal 98.5 % · tax 97.6 % · vendor 95.3 % · total 94.1 % |

Your job: confirm the app installs, runs, and behaves as described below, and try to break it.

---

## 1. Prerequisites

| Need | Version | Check |
|---|---|---|
| Python | 3.10 – 3.12 | `python --version` |
| Node.js | 18+ | `node --version` |
| Git | any | `git --version` |
| **Tesseract OCR** | 5.x | `tesseract --version` |
| Poppler *(only to test PDFs)* | any | optional |

* **Tesseract (Windows):** download the installer from <https://github.com/UB-Mannheim/tesseract/wiki> and keep the default folder
  (`C:\Program Files\Tesseract-OCR`). macOS: `brew install tesseract`. Linux: `sudo apt install tesseract-ocr`.
* **Poppler (PDF only):** Windows – download a build, unzip it, and note its `Library\bin` folder. macOS: `brew install poppler`. Linux: `sudo apt install poppler-utils`.
* **No GPU needed.** Everything runs on CPU. Disk: about 1 GB (mostly PyTorch).

---

## 2. Get the code and the two model files

```powershell
git clone https://github.com/im-rk/DocuSense.git
cd DocuSense          # the project root: the folder that contains api/, models/, frontend/ and samples/
```

### The model files are NOT in git (they are git-ignored, ~20 MB)
Ask the project owner for these two files and put them here:

```
models/checkpoints/classifier_v1.pth      (≈ 13 MB)
models/checkpoints/tagger_v1.pth          (≈ 7 MB)
```

> ✅ **Check:** both files exist in `models/checkpoints/`. Without them the app still starts, but every upload ends as FAILED (this is test T-12 below).

---

## 3. Install

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
If PowerShell blocks the activate script: `Set-ExecutionPolicy -Scope Process Bypass` and try again.

### macOS / Linux
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Tell the app where Tesseract is (only if `tesseract --version` fails in a new terminal)
Create a file named `.env` in the project root (copy `.env.example`) and add:
```
DOCUSENSE_TESSERACT_CMD=C:/Program Files/Tesseract-OCR/tesseract.exe
# only for PDFs:
# DOCUSENSE_POPPLER_PATH=C:/tools/poppler/Library/bin
```
Use forward slashes — backslashes get mangled in `.env` files.

### Frontend
```powershell
cd frontend
npm install
cd ..
```

> ✅ **Check:** `pip` and `npm install` finish without errors. (`npm` may print audit warnings; ignore them.)

---

## 4. Step 1 — Automated tests (2 minutes)

```powershell
pytest -q
```
**Expected:** `38 passed` (takes 1–2 minutes; a couple of harmless warnings are normal).
These cover: text/number parsing, reading order, dataset builders, model shapes and padding, a tiny training run, the API, the pipeline and the evaluation script.

> If you see failures, copy the **last 30 lines** of output into your report.

---

## 5. Step 2 — Run the pipeline from the command line (no web app yet)

Five sample invoices are in `samples/`. Run one:

```powershell
python -m pipeline.run samples/sample_1.png
```
**Expected:** JSON with `"document_type": "INVOICE"`, and `fields` equal to:

```
vendor: Zenith Services Corp   date: 2022-10-10   subtotal: 14103.83   tax: 1692.46   total: 15796.29
```
plus 7 line items, `"validation": {"passed": true, ...}`, and `inference_ms` of roughly 1000–3000.
(The invoice number may come back as `!N/2018-4082` instead of `IN/2018-4082` — that is Tesseract misreading a character. It is a **known** limitation, not a bug.)

If you get `Tesseract is not installed or not on PATH` → go back to section 3 (`.env`).

### Expected results for all five samples

| Sample | Vendor | Date | Subtotal | Tax | Total | Invoice no. | Model 1 says |
|---|---|---|---|---|---|---|---|
| sample_1 | Zenith Services Corp | 2022-10-10 | 14103.83 | 1692.46 | 15796.29 | `IN/2018-4082` (OCR may give `!N/…`) | INVOICE |
| sample_2 | Maple Bakery Corp | 2023-02-26 | 6801.31 | 544.10 | 7345.41 | `2018-2642` | RECEIPT (narrow receipt layout) |
| sample_3 | Urban Hardware Co. | 2025-05-23 | 25208.84 | 2016.71 | 27225.55 | `INV-2018-0009` | OTHER (see note) |
| sample_4 | Zenith Services LLP | 2021-08-25 | 16023.99 | 2884.32 | 18908.31 | `IN/2024-2661` (may be missing) | INVOICE |
| sample_5 | Star Supplies Inc | 2019-07-23 | 10803.37 | 1944.61 | 12747.98 | `INV-2023-6856` (OCR may give `\|NV-…`) | INVOICE |

* **Fields (vendor, date, subtotal, tax, total) should match exactly** on all five.
* **Invoice number is the weakest field** (about 40–60 % correct) — judge it leniently.
* **Document type:** these are synthetic images, so the page classifier (89.5 % overall accuracy) is allowed to be wrong on some, as with sample_3. Note it, don't fail it.

---

## 6. Step 3 — Start the backend

Open a terminal in the project root (venv active):

```powershell
uvicorn api.main:app --port 8000
```
**Expected:** `Uvicorn running on http://127.0.0.1:8000`.

> **Port 8000 busy?** (Docker/WSL often use it.) Use `--port 8010` and, when starting the frontend, set `VITE_API_URL=http://localhost:8010` (see next step).

Check in a browser:

| URL | Expected |
|---|---|
| <http://localhost:8000/health> | `{"status":"ok","models":{"loaded":false,"missing":[]}}` — **`missing` must be `[]`** (`loaded` turns `true` after the first document) |
| <http://localhost:8000/docs> | Swagger page listing `/documents`, `/documents/{id}`, `/documents/{id}/fields`, `/documents/{id}/export`, `/analytics/summary` |

---

## 7. Step 4 — Start the frontend

Second terminal:

```powershell
cd frontend
npm run dev
```
Using port 8010 for the backend? Start it like this instead:
* PowerShell: `$env:VITE_API_URL="http://localhost:8010"; npm run dev`
* macOS/Linux: `VITE_API_URL=http://localhost:8010 npm run dev`

Open **<http://localhost:5173>**.
**Expected:** a dark top bar ("DocuSense", tabs *Documents* / *Analytics*, a green dot with **"Models ready"**, a yellow **Upload** button) and three panes: document list (empty), page viewer, details.

---

## 8. Step 5 — Manual test checklist (the main part)

Tick each row. "Expected" is what a correct build does. Use the files in `samples/` unless stated.

### A. Upload and processing
| # | Action | Expected |
|---|---|---|
| T-01 | Click **Upload**, choose `samples/sample_1.png` | The document appears in the left list with a pulsing dot, then turns green within ~5 s. It auto-selects. The page shows with coloured highlights on words. |
| T-02 | Upload `sample_2.png`, `sample_3.png`, `sample_4.png`, `sample_5.png` **all at once** (multi-select) | 4 more rows appear; each finishes; none stays stuck on a pulsing dot or red. |
| T-03 | **Drag** a sample image from your file explorer onto the browser window | A yellow "Drop to upload" overlay appears while dragging; releasing uploads it. |
| T-04 | Right pane for `sample_1`: read **Fields** | Vendor, Date, Subtotal, Tax, Total match section 5. A small number at the right of each is the model's confidence (%). Line items table has 7 rows. |
| T-05 | Look at **Checks** (bottom of the right pane) | `PASS subtotal plus tax equals total`, `PASS items sum equals subtotal`, green "All checks passed." |

### B. Reviewing the page
| # | Action | Expected |
|---|---|---|
| T-06 | Hover (or click into) the **Total** field | On the page, only the total amount stays highlighted; all other highlights fade. Move to **Vendor** → only the vendor name stays lit. |
| T-07 | Click a colour chip in the legend (e.g. "item price") | Those highlights hide; click again, they return. |
| T-08 | Use **−**, **Fit**, **+** above the page | The page zooms out / fits the pane / zooms in. |
| T-09 | Hover a highlighted word on the page | Tooltip shows the word, its predicted label and probability. |

### C. Editing and saving
| # | Action | Expected |
|---|---|---|
| T-10 | Change **Vendor** to `Test Co`. | The field turns yellow; **Save changes** becomes enabled. |
| T-11 | Press **Ctrl+S** (or click Save) | A green "Saved" appears. Refresh the browser (F5), re-select the document: vendor is still `Test Co`. The list row now shows `Test Co`. |
| T-12 | Change **Total** to `1.00` and save | Status changes to *Needs review*. **Checks** shows `FAIL subtotal plus tax equals total` and a red message like *"subtotal + tax (…) != total (1.00)"*. Put the real total back and save: the check returns to PASS. |
| T-13 | **Add row** under Line items, type a name, then click **×** on it | A row is added then removed. Save is enabled while the table differs from the original. |

### D. Export
| # | Action | Expected |
|---|---|---|
| T-14 | Click **json**, **csv**, **xlsx** (top of right pane) | Three files download. JSON has `fields`, `line_items`, `validation`. CSV opens in Excel with one row per line item. XLSX has two sheets, *fields* and *line_items*. |

### E. List and navigation
| # | Action | Expected |
|---|---|---|
| T-15 | Click into the page (not a text box) and press **↓ / ↑** (or **J / K**) | Selection moves through the list and the page and fields update. |
| T-16 | Type `Maple` in the search box | List filters to matching rows (searches name, vendor, type). Type `zzzz` → "Nothing matches". Clear the box. |
| T-17 | Hover a list row and click the **×** | A confirmation asks "Delete …?"; confirming removes it (also removes it from Analytics). |

### F. Analytics
| # | Action | Expected |
|---|---|---|
| T-18 | Click the **Analytics** tab | Four numbers (Documents, Need review, Corrected by hand, ms per document), a "Documents per day" bar chart with today's bar, status counts, and average-confidence bars per field. |
| T-19 | After T-10/T-11 you corrected a document | "Corrected by hand" is greater than 0 %. |

### G. Things that should be refused or fail gracefully (try to break it)
| # | Action | Expected |
|---|---|---|
| T-20 | Upload a `.txt` file (any text file) | A red message at the bottom: *Unsupported file type '.txt'. Allowed: …*. Nothing is added to the list. |
| T-21 | Upload a **blank white image** (screenshot an empty window, or make one in Paint) | Document is processed but ends as *Needs review* with errors *"total amount not found"* and *"no text found on the page"*. App does not crash. |
| T-22 | Rename a text file to `fake.png` and upload it | Document ends as *Failed* with the message *"The file is not a readable image or PDF …"*. |
| T-23 | Upload a file larger than 20 MB | Rejected with *File larger than 20 MB* (HTTP 413). |
| T-24 | Upload a **PDF** *(needs Poppler, section 1)* | First page is read and processed like an image. Without Poppler: *Failed* with a Poppler error — expected. Only the **first page** is read. |
| T-25 | **Missing models:** stop the backend, rename `models/checkpoints/tagger_v1.pth` to `x.pth`, restart, reload the page | The top bar shows a red dot and *"Missing: models/checkpoints/tagger_v1.pth"*. An upload ends as *Failed* with *"Trained model files not found … copy the .pth files into models/checkpoints/"*. **Rename it back** and restart. |
| T-26 | **Backend off:** stop the backend while the UI is open, then reload | Top bar shows *"API not reachable"* (red dot). No blank white screen. |

---

## 9. Step 6 — API check without the UI (optional, 5 minutes)

With the backend running (use the Swagger page at `/docs`, or curl):

```bash
# 1. upload
curl -F "file=@samples/sample_2.png" http://localhost:8000/documents
#    -> {"document_id":"<ID>","status":"PENDING"}   (HTTP 202)

# 2. poll until status is COMPLETE / NEEDS_REVIEW
curl http://localhost:8000/documents/<ID>

# 3. correct a field
curl -X PUT http://localhost:8000/documents/<ID>/fields -H "Content-Type: application/json" -d "{\"fields\":{\"vendor\":\"Test Co\"}}"

# 4. exports
curl "http://localhost:8000/documents/<ID>/export?fmt=csv"

# 5. summary
curl http://localhost:8000/analytics/summary
```
**Expected edge responses:** unknown field name in step 3 → `422`; unknown document id → `404`; `.txt` upload → `415`; empty file → `400`.

---

## 10. Step 7 — Accuracy check with the evaluation script

This is the same script used to measure real-world accuracy. Run it on the samples first:

```powershell
python -m evaluation.run_evaluation evaluation/own_eval.samples.json
```
**Expected:** per-file lines, then something like
`FIELD ACCURACY: vendor 1.0, date 1.0, subtotal 1.0, tax 1.0, total 1.0, invoice_number ≈ 0.4 | overall ≈ 0.9 | avg ≈ 1–2 s/doc`.
A result file is written to `evaluation/results.json`.

### Strongly recommended: test on REAL documents
The samples are computer-generated, so the numbers above are optimistic. Real phone photos are the honest test.
1. Put 10–30 real receipts/invoices (photos, scans) in a folder, e.g. `evaluation/docs/`.
2. Copy `evaluation/own_eval.example.json` to `evaluation/own_eval.json` and list each file with the **true** values you can read off it (only the fields you want scored).
3. `python -m evaluation.run_evaluation`
4. Report the field accuracies. **Expect lower than the samples** (shadows, creases, handwriting, different layouts and currencies hurt OCR).
   For reference, receipts from the training style (Malaysian/Indonesian, printed thermal) do best.

---

## 11. Known limitations (do not report these as bugs)

* **Tesseract is the only non-from-scratch part.** OCR mistakes (`I`→`!`, `0`→`O`) flow into the result; the correction feature exists for this.
* **Invoice number** is the weakest field (OCR character confusion + few training examples).
* **Page classifier accuracy is 89.5 %**, mainly INVOICE ↔ FORM confusion; the target was 93 % and was not reached.
* Trained on printed receipts (CORD, SROIE) plus synthetic invoices: handwriting, non-English text and unusual layouts are weaker.
* **Only the first page** of a PDF is processed. Whole-page photos work best (skewed, cropped or very dark photos are weaker).
* The receipt class comes from different datasets than the other four page types, so receipt recognition on very unusual photos is worth testing separately.
* No login/accounts (single-user local tool).

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| `Tesseract is not installed or not on PATH` | Install it, open a **new** terminal, or set `DOCUSENSE_TESSERACT_CMD` in `.env` (forward slashes). |
| `/health` shows `"missing": ["models/checkpoints/…"]` | Put `classifier_v1.pth` and `tagger_v1.pth` in `models/checkpoints/` and restart the backend. |
| UI shows "API not reachable" | Backend not running, or on a different port. Match `VITE_API_URL` to the backend port. |
| Port 8000 already in use | `uvicorn api.main:app --port 8010` and `VITE_API_URL=http://localhost:8010`. |
| Browser console shows CORS errors | The UI must be opened at `localhost:5173` or `127.0.0.1:5173` (the allowed origins). |
| `numpy` / `torch` import errors | Use the pinned versions: `pip install torch==2.3.1 ...` then `pip install -r requirements.txt` (numpy must be < 2). |
| `ModuleNotFoundError: No module named 'api'` (or `models`, `data`) | Run commands from the **project root**, not from inside a sub-folder. |
| PDF upload fails | Install Poppler and set `DOCUSENSE_POPPLER_PATH` (section 3). |
| Upload stuck on "PENDING" forever | Look at the terminal running uvicorn for the error. The first request takes a few extra seconds while models load. |
| Want a clean slate | Stop the backend, delete `docusense.db` and the `uploads/` folder, start again. |

---

## 13. Where things are (to trace the flow in the code)

| Stage | File |
|---|---|
| Upload endpoint, list, edit, export | `api/routers/documents.py` |
| Background job | `api/tasks.py` → `pipeline/run.py` (`process`) |
| Page → image (PDF/EXIF) | `pipeline/preprocess.py` |
| **Model 1** architecture / inference | `models/classifier/architecture.py`, `models/classifier/predict.py` |
| OCR | `pipeline/ocr.py` |
| **Model 2** architecture / loss / inference | `models/tagger/architecture.py`, `models/tagger/loss.py` (partial-label loss), `models/tagger/predict.py` |
| Turning labels into fields, dates, amounts | `models/tagger/postprocess.py` |
| Arithmetic and confidence checks | `validation/rules.py` |
| Training (run on Kaggle) | `notebooks/kaggle_1…4_*.ipynb`, `KAGGLE_TRAINING.md`, `models/*/train.py` |
| UI | `frontend/src/pages/Workspace.tsx`, `frontend/src/components/` |

---

## 14. What to send back

Copy this and fill it in:

```
Tester: ____________   Date: ________   OS: ________   Python: ____   Node: ____
Tesseract version: ____   Poppler installed: yes / no

pytest:              passed ___ / failed ___
CLI sample_1:        fields correct? yes / no
Backend /health:     missing = []  yes / no
UI checklist:        T-01 … T-26   -> list any that FAILED, with what you saw
Evaluation (samples): overall ___
Evaluation (own docs, n=___): vendor ___  date ___  total ___  tax ___  invoice_no ___
Anything confusing / crashes / screenshots: ____________________
```

**The project passes if:** pytest is green, `/health` shows no missing models, samples 1–5 return the correct vendor/date/subtotal/tax/total, the UI checklist rows T-01 … T-26 behave as described, and nothing crashes the app.

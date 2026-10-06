# DocuSense

Receipt & invoice understanding with **two deep-learning models trained completely from scratch** (no pretrained weights,
no LLM/API): upload a scan, photo or PDF → get validated, structured data (vendor, date, subtotal, tax, total, line items).

```
image / PDF ─► Model 1  DocClassifierResNet   page type: RECEIPT | INVOICE | FORM | LETTER | OTHER
            ─► OCR (Tesseract)                 words + boxes        (the only non-from-scratch part, isolated in pipeline/ocr.py)
            ─► Model 2  DocTaggerNet           label every word: VENDOR DATE ITEM_* SUBTOTAL TAX TOTAL INVOICE_NO
            ─► post-process + validation       fields, line items, arithmetic checks, confidences
            ─► FastAPI + SQLite ─► React UI (page with coloured word boxes, editable fields, export)
```

| Model | Architecture | Params | Data |
|---|---|---|---|
| 1 · page classifier | ResNet (8 residual blocks, 1×256×256 grayscale) | ~3.5 M | RVL-CDIP subset (streamed) + CORD/SROIE receipts |
| 2 · word tagger | per word: **text-CNN** (chars) + **look-CNN** (word crop) + **position MLP** → 2-layer **BiLSTM** → label | ~2 M | CORD-v2 + SROIE (+ synthetic invoices) with a **partial-label loss** |

Training is done on **Kaggle** (see [KAGGLE_TRAINING.md](KAGGLE_TRAINING.md)); the trained `.pth` files are copied into
`models/checkpoints/` and used here on CPU. Design docs: `../MD's/` (PRD, plan, review of the old design).

## Run it (Windows)
```powershell
python -m venv .venv ; .venv\Scripts\Activate.ps1
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# also install Tesseract (UB-Mannheim build) and, for PDFs, Poppler; see .env.example if they are not on PATH

# after copying classifier_v1.pth + tagger_v1.pth into models/checkpoints/ :
python -m pipeline.run path\to\receipt.jpg          # CLI: prints the JSON
uvicorn api.main:app --reload                       # API + Swagger at http://localhost:8000/docs
cd frontend ; npm install ; npm run dev             # UI at http://localhost:5173
```
Without the checkpoints the API still starts; uploads are marked FAILED with a message that says what to do.

> **Testing the whole project end to end?** Follow [TESTING.md](TESTING.md) (setup, sample files, manual checklist).

## Tests
```powershell
pytest -q            # parsing, validation, dataset builders, model shapes/padding/parity, tiny CPU training, API, pipeline
```
The model tests train tiny models for a couple of epochs on synthetic data just to prove the code is correct;
real training is on Kaggle.

## Layout
```
models/classifier/  architecture · dataset · train · evaluate · predict · preprocess     (Model 1)
models/tagger/      architecture · dataset · loss · train · evaluate · predict · postprocess · baseline · features · vocab  (Model 2)
data/               build_cord · build_sroie · build_rvl_cache · synth_invoices · reading_order · crops · viz_labels
notebooks/          kaggle_1..4 (.py with # %% cells, and generated .ipynb to import into Kaggle)
pipeline/           preprocess · ocr · loader · run     validation/rules.py     evaluation/run_evaluation.py
api/  db/  storage/  config/  frontend/  tools/  tests/     _old_v1/ = the previous design (kept for reference only)
```

## Results
Measured on Kaggle (T4), held-out test splits. Field accuracy uses the datasets' own words ("GT OCR"), not Tesseract.

| | Value |
|---|---|
| Model 1 test accuracy (5 classes, 3.3 M params) | **89.5 %** (target was 93 %, not reached). Recall: RECEIPT 98.9 · INVOICE 84.6 · FORM 87.6 · LETTER 88.6 · OTHER 88.8 |
| Model 2 token macro-F1 (CORD + SROIE test, 1.79 M params) | **0.970** (rule-based baseline: 0.267) |
| Field accuracy (GT OCR) | date 100 % · subtotal 98.5 % · tax 97.6 % · vendor 95.3 % · total 94.1 % |
| Live pipeline (real Tesseract + both models) on 20 unseen synthetic invoices | vendor 20/20 · subtotal 20/20 · total 20/20 · date 19/20 · tax 19/20 · invoice no. 11/20 (OCR confuses `I`/`!`, `0`/`O`) |
| Field accuracy on own real documents (live OCR) | _to measure_ (`python -m evaluation.run_evaluation`) |

Curves, confusion matrix and reports are in `results/` (git-ignored). The app was also checked end to end (upload, background
processing, validation, export JSON/CSV/XLSX, field edit, analytics) with the trained checkpoints and OCR stubbed.

## Limitations (honest)
Training data are receipts (CORD Indonesian, SROIE Malaysian) + synthetic invoices; real invoices from other layouts,
languages or currencies will be weaker. OCR errors propagate (Tesseract is external and pretrained). Only the first page of a PDF is processed.

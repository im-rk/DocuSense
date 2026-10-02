"""Measure the REAL pipeline (live Tesseract OCR + both models) on your own hand-annotated documents.

1. Put 20-30 receipts/invoices (photos, scans, PDFs) in evaluation/docs/
2. Copy evaluation/own_eval.example.json -> evaluation/own_eval.json and fill in the true values you can read off each document
   (leave out a field you don't want scored).
3. python -m evaluation.run_evaluation            (needs the trained checkpoints, Tesseract)

Output: per-field accuracy + latency, saved to evaluation/results.json. These are the numbers for your report/README.
"""
import json
import os
import re
import sys
import time

from models.tagger.postprocess import normalise_date
from pipeline.run import process

FIELDS = ("vendor", "date", "invoice_number", "subtotal", "tax", "total")


def _same(field, gt, pred):
    if pred is None:
        return False
    if field in ("vendor", "invoice_number"):
        n = lambda s: re.sub(r"[^A-Z0-9]", "", str(s).upper())
        return n(gt) == n(pred)
    if field == "date":
        return (normalise_date(str(gt)) or str(gt)) == pred
    return abs(float(gt) - float(pred)) < 0.01


def main(path="evaluation/own_eval.json"):
    items = json.load(open(path, encoding="utf-8"))
    hit, tot, rows, times = {f: 0 for f in FIELDS}, {f: 0 for f in FIELDS}, [], []
    for it in items:
        t0 = time.time()
        res = process(it["file"])
        times.append(time.time() - t0)
        row = {"file": it["file"], "type": res["document_type"], "status": "REVIEW" if res["validation"]["needs_review"] else "OK"}
        for f, gt in it["ground_truth"].items():
            if f not in FIELDS:
                continue
            ok = _same(f, gt, res["fields"].get(f))
            tot[f] += 1
            hit[f] += int(ok)
            row[f] = {"gt": gt, "pred": res["fields"].get(f), "ok": ok}
        rows.append(row)
        print(row["file"], {k: v["ok"] for k, v in row.items() if isinstance(v, dict)})
    acc = {f: round(hit[f] / tot[f], 3) for f in FIELDS if tot[f]}
    overall = round(sum(hit.values()) / max(sum(tot.values()), 1), 3)
    out = {"n_documents": len(items), "field_accuracy": acc, "overall": overall,
           "avg_seconds_per_document": round(sum(times) / max(len(times), 1), 2), "rows": rows}
    json.dump(out, open("evaluation/results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("\nFIELD ACCURACY:", acc, "| overall:", overall, "| avg s/doc:", out["avg_seconds_per_document"])


if __name__ == "__main__":
    main(*(sys.argv[1:2]))

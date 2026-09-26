"""Shared writer for the word-tagging datasets (CORD, SROIE, synthetic).

Output per split name `X`:
    X.jsonl        one document per line (words already in reading order, with labels and line flags)
    X_crops.npy    uint8 (total_words, 32, 96); a document's crops start at record["crop_offset"]
    X_receipt.npy  uint8 (n_docs, 256, 256) letterboxed page (only used to give Model 1 its RECEIPT class)
"""
import json
import os
from collections import Counter

import numpy as np

from data.crops import make_crops
from data.reading_order import order_words
from data.viz_labels import draw_words
from models.classifier.preprocess import letterbox_gray
from models.tagger.features import MAX_WORDS
from models.tagger.labels import LABELS


class SplitBuilder:
    def __init__(self, out_dir: str, name: str, audit: int = 0, keep_receipt: bool = True):
        self.out_dir, self.name, self.audit, self.keep_receipt = out_dir, name, audit, keep_receipt
        self.records, self.crops, self.receipts = [], [], []
        self.offset = 0
        self.label_counts = Counter()
        os.makedirs(out_dir, exist_ok=True)

    def add(self, doc_id: str, source: str, split: str, gray: np.ndarray, words: list, pil_img=None) -> bool:
        """gray: uint8 (H,W). words: dicts {text, box(px), label(int)} in ANY order. Returns False if empty."""
        ordered = order_words(words)[:MAX_WORDS]
        if not ordered:
            return False
        H, W = gray.shape
        self.crops.append(make_crops(gray, [w["box"] for w in ordered]))
        self.records.append({
            "id": doc_id, "source": source, "split": split, "width": W, "height": H,
            "crop_offset": self.offset,
            "words": [{"text": w["text"], "box": [round(float(v), 1) for v in w["box"]], "label": int(w["label"]),
                       "line": w["line"], "is_first": w["is_first"], "is_last": w["is_last"]} for w in ordered]})
        self.offset += len(ordered)
        self.label_counts.update(int(w["label"]) for w in ordered)
        if self.keep_receipt and pil_img is not None:
            self.receipts.append(letterbox_gray(pil_img))
        if len(self.records) <= self.audit and pil_img is not None:
            adir = os.path.join(self.out_dir, "audit")
            os.makedirs(adir, exist_ok=True)
            draw_words(pil_img, ordered).save(os.path.join(adir, f"{doc_id}.jpg"), quality=80)
        return True

    def save(self) -> dict:
        p = lambda suffix: os.path.join(self.out_dir, f"{self.name}{suffix}")
        with open(p(".jsonl"), "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        np.save(p("_crops.npy"), np.concatenate(self.crops) if self.crops else np.zeros((0, 32, 96), np.uint8))
        if self.keep_receipt:
            np.save(p("_receipt.npy"), np.stack(self.receipts) if self.receipts
                    else np.zeros((0, 256, 256), np.uint8))
        return {"name": self.name, "docs": len(self.records), "words": self.offset,
                "labels": {LABELS[k]: v for k, v in sorted(self.label_counts.items())}}


def read_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

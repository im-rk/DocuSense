"""Model 2 dataset: unified jsonl + crop cache (built by data/build_*.py) -> tensors."""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from models.tagger.features import CROP_H, CROP_W, MAX_CHARS, crops_to_float, encode_chars, pos_features
from models.tagger.labels import N_CLS, unannot_mask

_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def load_records(data_dir: str, name: str):
    """-> (records, crops memmap)"""
    with open(os.path.join(data_dir, f"{name}.jsonl"), encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    return recs, np.load(os.path.join(data_dir, f"{name}_crops.npy"), mmap_mode="r")


class TaggerDataset(Dataset):
    def __init__(self, data_dir, names, vocab, augment=False, limits=None):
        """names: split names present in data_dir. limits: {name: max_docs} (used to cap synthetic data)."""
        self.vocab, self.augment, self._rng = vocab, augment, None
        self.docs = []                                                   # (record, crops_memmap)
        for name in names:
            recs, crops = load_records(data_dir, name)
            if limits and name in limits:
                recs = recs[: limits[name]]
            self.docs += [(r, crops) for r in recs]

    def __len__(self):
        return len(self.docs)

    @property
    def records(self):
        return [r for r, _ in self.docs]

    def _rng_(self):
        if self._rng is None:
            self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._rng

    def __getitem__(self, i):
        rec, crops = self.docs[i]
        words, n = rec["words"], len(rec["words"])
        boxes = np.array([w["box"] for w in words], dtype=np.float32)
        texts = [w["text"] for w in words]
        labels = np.array([w["label"] for w in words], dtype=np.int64)
        first = np.array([w["is_first"] for w in words], dtype=np.float32)
        last = np.array([w["is_last"] for w in words], dtype=np.float32)
        cr = np.array(crops[rec["crop_offset"]: rec["crop_offset"] + n])
        W, H = rec["width"], rec["height"]
        x_crops = crops_to_float(cr)
        if self.augment:
            boxes, texts, labels, first, last, x_crops = self._augment(boxes, texts, labels, first, last, x_crops, W, H)
        return {"chars": torch.from_numpy(encode_chars(texts, self.vocab)),
                "crops": torch.from_numpy(np.ascontiguousarray(x_crops)),
                "pos": torch.from_numpy(pos_features(boxes, W, H, first, last)),
                "labels": torch.from_numpy(labels),
                "unannot": torch.tensor(unannot_mask(rec["source"]), dtype=torch.bool),
                "id": rec["id"], "source": rec["source"]}

    def _augment(self, boxes, texts, labels, first, last, x_crops, W, H):
        r, n = self._rng_(), len(texts)
        # word dropout (5%), always keep a few words
        keep = r.random(n) > 0.05
        if keep.sum() < min(4, n):
            keep[:] = True
        boxes, labels, first, last, x_crops = boxes[keep], labels[keep], first[keep], last[keep], x_crops[keep]
        texts = [t for t, k in zip(texts, keep) if k]
        n = len(texts)
        # global scale/shift of the whole layout + small per-word jitter
        s = r.uniform(0.95, 1.05)
        c = np.array([W / 2, H / 2, W / 2, H / 2], dtype=np.float32)
        shift = np.array([r.uniform(-0.02, 0.02) * W, r.uniform(-0.02, 0.02) * H] * 2, dtype=np.float32)
        jit = np.stack([r.uniform(-0.005, 0.005, n) * W, r.uniform(-0.005, 0.005, n) * H] * 2, axis=1).astype(np.float32)
        boxes = (boxes - c) * s + c + shift + jit
        # character noise: drop or substitute one character in 15% of words
        out = []
        for t in texts:
            if len(t) > 1 and r.random() < 0.15:
                j = int(r.integers(len(t)))
                t = t[:j] + t[j + 1:] if r.random() < 0.5 else t[:j] + _CHARS[int(r.integers(len(_CHARS)))] + t[j + 1:]
            out.append(t)
        # crop appearance: per-word contrast/brightness, one global horizontal shift of up to 2 px
        a = r.uniform(0.8, 1.2, (n, 1, 1, 1)).astype(np.float32)
        b = r.uniform(-0.1, 0.1, (n, 1, 1, 1)).astype(np.float32)
        x_crops = np.clip(x_crops * a + b, 0.0, 1.0)
        x_crops = np.roll(x_crops, int(r.integers(-2, 3)), axis=3)
        return boxes, out, labels, first, last, x_crops


def collate(batch):
    B, N = len(batch), max(len(b["labels"]) for b in batch)
    chars = torch.zeros(B, N, MAX_CHARS, dtype=torch.long)
    crops = torch.zeros(B, N, 1, CROP_H, CROP_W)
    pos = torch.zeros(B, N, batch[0]["pos"].shape[1])
    labels = torch.zeros(B, N, dtype=torch.long)
    pad = torch.ones(B, N, dtype=torch.bool)
    for i, b in enumerate(batch):
        n = len(b["labels"])
        chars[i, :n], crops[i, :n], pos[i, :n], labels[i, :n], pad[i, :n] = b["chars"], b["crops"], b["pos"], b["labels"], False
    return {"chars": chars, "crops": crops, "pos": pos, "labels": labels, "pad": pad,
            "unannot": torch.stack([b["unannot"] for b in batch]),
            "ids": [b["id"] for b in batch], "sources": [b["source"] for b in batch]}


def label_counts(records) -> np.ndarray:
    c = np.zeros(N_CLS, dtype=np.int64)
    for r in records:
        for w in r["words"]:
            c[w["label"]] += 1
    return c

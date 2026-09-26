"""Model 1 data: cache 256x256 uint8 grayscale pages for RECEIPT / INVOICE / FORM / LETTER / OTHER.

RECEIPT comes from the CORD + SROIE pages saved by build_cord / build_sroie (`*_receipt.npy`).
The other four classes come from RVL-CDIP, STREAMED with a per-class cap (the full set is ~38 GB - never download it):
    INVOICE, FORM, LETTER, OTHER = {email, memo, resume, news article} (sampled evenly)

Classes are mapped BY NAME from the dataset's own label names, never by hard-coded ids: different Hugging Face copies use
different id orders (the original uses letter=0 ... invoice=11; alphabetical copies use invoice=6).
"""
import json
import math
import os

import numpy as np
from PIL import Image

from models.classifier.preprocess import letterbox_gray
from models.tagger.labels import DOC_CLASSES

CID = {n: i for i, n in enumerate(DOC_CLASSES)}
NAME_TO_OURS = {"invoice": "INVOICE", "form": "FORM", "letter": "LETTER",
                "email": "OTHER", "memo": "OTHER", "resume": "OTHER", "news article": "OTHER"}
# label order of the ORIGINAL RVL-CDIP release (used only if a dataset exposes no label names)
ORIGINAL_NAMES = ["letter", "form", "email", "handwritten", "advertisement", "scientific report",
                  "scientific publication", "specification", "file folder", "news article", "budget", "invoice",
                  "presentation", "questionnaire", "resume", "memo"]
DEFAULT_SOURCES = ("chainyo/rvl-cdip", "aharley/rvl_cdip")


def _key(n: str) -> str:
    return str(n).lower().replace("_", " ").replace("-", " ").strip()


def _targets(names, quota):
    """label id -> (our class name, how many to collect)."""
    per_other = math.ceil(quota / 4)
    out = {}
    for i, n in enumerate(names):
        ours = NAME_TO_OURS.get(_key(n))
        if ours:
            out[i] = (ours, per_other if ours == "OTHER" else quota)
    return out


def collect_hf(hf_id, hf_split, quota, seed=0, buffer_size=5000, max_seen=400_000):
    from datasets import load_dataset
    kw = {"trust_remote_code": True} if hf_id == "aharley/rvl_cdip" else {}      # script-based repo (needs datasets<4)
    ds = load_dataset(hf_id, split=hf_split, streaming=True, **kw)
    feat = getattr(ds, "features", None)
    names = feat["label"].names if feat and "label" in feat else None
    if names is None:
        print(f"  WARNING: {hf_id} exposes no label names - assuming the original RVL-CDIP id order")
        names = ORIGINAL_NAMES
    target = _targets(names, quota)
    print(f"  {hf_id}/{hf_split}: label map {({names[i]: t[0] for i, t in target.items()})}")
    have = {i: 0 for i in target}
    X, Y = [], []
    for seen, ex in enumerate(ds.shuffle(seed=seed, buffer_size=buffer_size)):
        lab = int(ex["label"])
        if lab in target and have[lab] < target[lab][1]:
            X.append(letterbox_gray(ex["image"])); Y.append(CID[target[lab][0]]); have[lab] += 1
        if seen % 4000 == 0:
            print(f"  {hf_split}: seen {seen}, collected {sum(have.values())}/{sum(t[1] for t in target.values())}")
        if all(have[i] >= target[i][1] for i in target) or seen > max_seen:
            break
    short = {names[i]: (have[i], target[i][1]) for i in target if have[i] < target[i][1]}
    if short:
        print(f"  NOTE: fewer images than requested for {short}")
    return X, Y


def collect_dir(root, quota, seed=0):
    """Fallback: a local folder with one sub-folder per class, named by class name (e.g. 'invoice', 'news_article')."""
    rng = np.random.default_rng(seed)
    per_other = math.ceil(quota / 4)
    X, Y = [], []
    for sub in sorted(os.listdir(root)):
        ours = NAME_TO_OURS.get(_key(ORIGINAL_NAMES[int(sub)] if sub.isdigit() else sub))
        if not ours:
            continue
        files = [f for f in os.listdir(os.path.join(root, sub)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]
        rng.shuffle(files)
        for f in files[: (per_other if ours == "OTHER" else quota)]:
            X.append(letterbox_gray(Image.open(os.path.join(root, sub, f)))); Y.append(CID[ours])
    return X, Y


def build_cls_cache(out_dir, quotas=None, hf_ids=DEFAULT_SOURCES, rvl_dir=None, seed=0):
    quotas = quotas or {"train": 3000, "val": 500, "test": 500}
    hf_split = {"train": "train", "val": "validation", "test": "test"}
    rng = np.random.default_rng(seed)
    summary, stats = {}, {}
    for split in ("train", "val", "test"):
        X, Y, err = [], [], []
        if rvl_dir:
            X, Y = collect_dir(rvl_dir, quotas[split], seed)
        else:
            for hf_id in hf_ids:                                    # first source that works wins
                try:
                    X, Y = collect_hf(hf_id, hf_split[split], quotas[split], seed)
                    if X:
                        break
                except Exception as e:                              # dataset id/format changed, no network, ...
                    err.append(f"{hf_id}: {type(e).__name__}: {str(e)[:200]}")
                    print("  source failed ->", err[-1])
        if not X:
            raise RuntimeError("Could not collect RVL-CDIP images. Tried: " + " | ".join(err) +
                               "\nFix: attach an RVL-CDIP copy and pass rvl_dir=<folder with one sub-folder per class>.")
        for src in ("cord", "sroie"):
            p = os.path.join(out_dir, f"{src}_{split}_receipt.npy")
            if os.path.exists(p):
                r = np.load(p)
                X += list(r); Y += [CID["RECEIPT"]] * len(r)
        X, Y = np.stack(X).astype(np.uint8), np.array(Y, dtype=np.int64)
        order = rng.permutation(len(Y))
        X, Y = X[order], Y[order]
        np.save(os.path.join(out_dir, f"cls_{split}_x.npy"), X)
        np.save(os.path.join(out_dir, f"cls_{split}_y.npy"), Y)
        summary[split] = {c: int((Y == i).sum()) for c, i in CID.items()}
        print(split, summary[split])
        if split == "train":
            v = X.astype(np.float32) / 255.0
            stats = {"mean": float(v.mean()), "std": float(v.std())}
            del v
    stats["counts"] = summary
    json.dump(stats, open(os.path.join(out_dir, "cls_stats.json"), "w"), indent=1)
    print(stats)
    return stats

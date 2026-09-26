"""CORD-v2 (Hugging Face `naver-clova-ix/cord-v2`, CC-BY-4.0) -> unified word-level dataset.

CORD annotates items, subtotal, tax and total at word level in ground_truth["valid_line"].
It has NO vendor / date / invoice-number labels (those come from SROIE) - see partial-label loss.
"""
import json

import numpy as np

from data.build_common import SplitBuilder
from models.tagger.labels import LABEL_ID as L

CORD_MAP = {
    "menu.nm": L["ITEM_NAME"], "menu.sub_nm": L["ITEM_NAME"],
    "menu.cnt": L["ITEM_QTY"], "menu.price": L["ITEM_PRICE"],
    "sub_total.subtotal_price": L["SUBTOTAL"], "sub_total.tax_price": L["TAX"],
    "total.total_price": L["TOTAL"],
}   # everything else (unit price, cash, change, discounts, void items, keys ...) -> OTHER


def _box(quad, sx, sy):
    if isinstance(quad, dict):
        xs = [quad[f"x{i}"] for i in (1, 2, 3, 4)]
        ys = [quad[f"y{i}"] for i in (1, 2, 3, 4)]
    else:                                         # [x1,y1,...,x4,y4]
        xs, ys = quad[0::2], quad[1::2]
    return [min(xs) * sx, min(ys) * sy, max(xs) * sx, max(ys) * sy]


def cord_words(sample):
    """HF sample -> (PIL RGB image, gray uint8, words[{text, box, label}])."""
    img = sample["image"].convert("RGB")
    gray = np.asarray(img.convert("L"))
    gt = json.loads(sample["ground_truth"]) if isinstance(sample["ground_truth"], str) else sample["ground_truth"]
    W, H = img.size
    meta = gt.get("meta", {}).get("image_size", {})
    sx, sy = W / float(meta.get("width", W)), H / float(meta.get("height", H))
    words = []
    for line in gt.get("valid_line", []):
        label = CORD_MAP.get(line.get("category", ""), 0)
        for w in line.get("words", []):
            text = str(w.get("text", "")).strip()
            if not text:
                continue
            lab = 0 if w.get("is_key", 0) == 1 else label      # key text ("TOTAL:") is not the value
            words.append({"text": text, "box": _box(w["quad"], sx, sy), "label": lab})
    return img, gray, words


def build_cord(out_dir: str, hf_id: str = "naver-clova-ix/cord-v2", audit: int = 20, limit: int = None) -> list:
    from datasets import load_dataset
    stats = []
    for hf_split, name in (("train", "cord_train"), ("validation", "cord_val"), ("test", "cord_test")):
        ds = load_dataset(hf_id, split=hf_split)
        sb = SplitBuilder(out_dir, name, audit=audit)
        for i, sample in enumerate(ds):
            if limit and i >= limit:
                break
            img, gray, words = cord_words(sample)
            sb.add(f"{name}_{i:04d}", "cord", hf_split, gray, words, pil_img=img)
        stats.append(sb.save())
        print(stats[-1])
    return stats

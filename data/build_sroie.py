"""SROIE 2019 (ICDAR receipts) -> unified word-level dataset.

Expected on disk (Kaggle 'SROIE datasetv2' or the original ICDAR release; layout is auto-discovered):
    <split>/img/<id>.jpg
    <split>/box/<id>.txt        lines: x1,y1,x2,y2,x3,y3,x4,y4,TEXT      (line-level boxes)
    <split>/entities/<id>.txt   JSON: {"company":..., "date":..., "address":..., "total":...}
SROIE gives LINE boxes and only 4 key values, so:
  * lines are split into words (box split in proportion to characters),
  * VENDOR / DATE / TOTAL are labelled by matching the key values against the lines.
Documents whose key values cannot be located are DROPPED (their labels would be wrong) and counted.
"""
import difflib
import json
import os
import re
import zlib

import numpy as np
from PIL import Image

from data.build_common import SplitBuilder
from models.tagger.labels import LABEL_ID as L
from models.tagger.postprocess import parse_amount

IMG_EXT = (".jpg", ".jpeg", ".png")


def norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def parse_box_file(path: str) -> list:
    lines = []
    with open(path, encoding="utf-8-sig", errors="ignore") as f:
        for raw in f:
            parts = raw.strip().split(",", 8)
            if len(parts) < 9:
                continue
            try:
                c = [float(v) for v in parts[:8]]
            except ValueError:
                continue
            text = parts[8].strip()
            if text:
                lines.append({"text": text, "box": [min(c[0::2]), min(c[1::2]), max(c[0::2]), max(c[1::2])]})
    return lines


def split_line(text: str, box: list) -> list:
    """One line box -> per-word boxes, x split proportional to character position."""
    words, x0, y0, x1, y1 = text.split(), box[0], box[1], box[2], box[3]
    if len(words) <= 1:
        return [(text.strip(), list(box))]
    total, out, pos = max(len(text), 1), [], 0
    for w in words:
        start = text.index(w, pos)
        pos = start + len(w)
        out.append((w, [x0 + (x1 - x0) * start / total, y0, x0 + (x1 - x0) * pos / total, y1]))
    return out


def label_sroie(lines: list, ent: dict):
    """lines: [{text, box}] (any order). Returns (words[{text,box,label}], flags) or (None, flags) if a key is unfound."""
    lines = sorted(lines, key=lambda l: (l["box"][1] + l["box"][3]) / 2)
    words, line_of = [], []
    for li, l in enumerate(lines):
        for w, b in split_line(l["text"], l["box"]):
            words.append({"text": w, "box": b, "label": 0})
            line_of.append(li)
    flags = {"vendor": False, "date": False, "total": False}

    # VENDOR: word-level fuzzy match of the company name's words, searched only in the top lines of the receipt
    # (a vendor line often carries extra text such as a registration number, and OCR text has small errors).
    ctoks = [norm(t) for t in str(ent.get("company", "")).split() if norm(t)]
    if ctoks:
        found, hits = set(), []
        for i, (w, li) in enumerate(zip(words, line_of)):
            if li >= 10:                                        # vendor is at the top of a receipt
                continue
            nw = norm(w["text"])
            if not nw:
                continue
            for k, ct in enumerate(ctoks):
                if nw == ct or (len(nw) >= 4 and len(ct) >= 4 and difflib.SequenceMatcher(None, nw, ct).ratio() >= 0.8):
                    hits.append(i); found.add(k)
                    break
        flags["vendor"] = len(found) >= max(1, round(0.7 * len(ctoks)))
        if flags["vendor"]:
            vlines = {line_of[i] for i in hits}
            first = min(vlines)                                 # keep the contiguous block starting at the first hit line
            block = {first}
            while first + len(block) in vlines:
                block.add(first + len(block))
            for i in hits:
                if line_of[i] in block:
                    words[i]["label"] = L["VENDOR"]

    # DATE: first line containing the date string; label the words that are part of it
    dn = norm(ent.get("date", ""))
    if len(dn) >= 4:
        for li, l in enumerate(lines):
            if dn in norm(l["text"]):
                for w, lj in zip(words, line_of):
                    nw = norm(w["text"])
                    if lj == li and len(nw) >= 2 and (nw in dn or dn in nw):
                        w["label"] = L["DATE"]
                        flags["date"] = True
                if flags["date"]:
                    break

    # TOTAL: the word equal to the total; prefer a line that says TOTAL (not SUBTOTAL), last such occurrence
    tv = parse_amount(ent.get("total"))
    if tv is not None:
        cands = [i for i, w in enumerate(words)
                 if re.search(r"\d", w["text"]) and (p := parse_amount(w["text"])) is not None and abs(p - tv) < 0.006]
        def keyed(i):
            t = lines[line_of[i]]["text"].upper()
            return "TOTAL" in t and "SUB" not in t
        pref = [i for i in cands if keyed(i)]
        pick = (pref or cands or [None])[-1]
        if pick is not None:
            words[pick]["label"] = L["TOTAL"]
            flags["total"] = True
    # date + total are required; a receipt whose vendor cannot be located is kept, with VENDOR marked "not annotated"
    # (source 'sroie_nv'), so the partial-label loss does not teach the model that its vendor words are OTHER.
    return (words if flags["date"] and flags["total"] else None), flags


def find_items(root: str) -> list:
    """Auto-discover (image, box file, entities file, split) triples under `root`."""
    imgs, texts = {}, {}
    for dp, _, fs in os.walk(root):
        for f in fs:
            stem, ext = os.path.splitext(f)
            path = os.path.join(dp, f)
            if ext.lower() in IMG_EXT:
                imgs[stem] = path
            elif ext.lower() in (".txt", ".json"):
                texts.setdefault(stem, []).append(path)
    items = []
    for stem, ip in sorted(imgs.items()):
        box, ent = None, None
        for tp in texts.get(stem, []):
            with open(tp, encoding="utf-8-sig", errors="ignore") as f:
                content = f.read()
            if content.lstrip().startswith("{"):
                try:
                    ent = json.loads(content)
                except ValueError:
                    try:                                    # some copies store a Python-style dict (single quotes)
                        import ast
                        ent = ast.literal_eval(content)
                    except (ValueError, SyntaxError):
                        ent = None
            else:
                box = tp
        if box and ent:
            rel = os.path.relpath(ip, root).lower().replace("\\", "/").split("/")[:-1]     # only look BELOW the root
            split = "test" if any(re.fullmatch(r"test(ing|_set)?", p) for p in rel) else "train"
            items.append({"id": stem, "img": ip, "box": box, "ent": ent, "split": split})
    return items


def build_sroie(out_dir: str, root: str, val_frac: float = 0.10, audit: int = 20, limit: int = None) -> dict:
    items = find_items(root)
    if not items:
        seen = [os.path.relpath(os.path.join(dp, f), root) for dp, _, fs in os.walk(root) for f in fs][:15]
        raise FileNotFoundError(f"No SROIE (image + box txt + entities json) triples found under {root}. "
                                f"First files seen: {seen}. Send this list so the loader can be adapted.")
    builders = {"train": SplitBuilder(out_dir, "sroie_train", audit=audit),
                "val": SplitBuilder(out_dir, "sroie_val", audit=audit // 2),
                "test": SplitBuilder(out_dir, "sroie_test", audit=audit)}
    dropped = {"vendor": 0, "date": 0, "total": 0, "total_docs": 0, "seen": 0}
    no_vendor = 0
    for it in items[: limit or None]:
        dropped["seen"] += 1
        words, flags = label_sroie(parse_box_file(it["box"]), it["ent"])
        if words is None:
            dropped["total_docs"] += 1
            for k, v in flags.items():
                dropped[k] += int(not v)
            continue
        split = it["split"]
        if split == "train" and zlib.crc32(it["id"].encode()) % 1000 < val_frac * 1000:
            split = "val"                                            # deterministic 10% validation carve-out
        img = Image.open(it["img"]).convert("RGB")
        src = "sroie" if flags["vendor"] else "sroie_nv"             # vendor not located -> VENDOR unannotated
        no_vendor += int(not flags["vendor"])
        builders[split].add(f"sroie_{split}_{it['id']}", src, split, np.asarray(img.convert("L")), words, pil_img=img)
    stats = [b.save() for b in builders.values()]
    for s in stats:
        print(s)
    cov = 1 - dropped["total_docs"] / max(dropped["seen"], 1)
    print(f"SROIE label coverage: {cov:.1%} of documents kept (need date + total); drops by key: {dropped}")
    kept = dropped["seen"] - dropped["total_docs"]
    print(f"SROIE vendor labelled on {1 - no_vendor / max(kept, 1):.1%} of kept documents "
          f"({no_vendor} kept with VENDOR marked 'not annotated')")
    if cov < 0.95:
        print("WARNING: coverage below the 95% gate - inspect the audit images and fix the matching rules.")
    return {"splits": stats, "dropped": dropped, "coverage": cov, "no_vendor": no_vendor}
